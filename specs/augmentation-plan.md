# Augmentation Implementation Plan

## Summary

Implement the augmentation system described in `specs/augmentation.md`. This adds `augment_<field_name>` methods to HyraxDataset classes, a new `augment` config key on data_request entries, modified data dispatch in DataProvider, and an `on_epoch_start` lifecycle callback. The existing DataCache continues to cache `get_<field>` results; `augment_<field>` is applied post-cache and its results are not cached in V1.

## Scope

**V1 only** — `augment` is a boolean. V2 (per-field dict) is out of scope but the design should not preclude it.

---

## Step 1: Config Schema — Add `augment` field to `DataRequestConfig`

**File:** `src/hyrax/config_schemas/data_request.py`

Add to `DataRequestConfig`:

```python
augment: bool | None = Field(
    None,
    description="Enable augmentation for this dataset. When true, augment_<field> methods are used.",
)
```

Add a model validator that rejects `augment=True` on data groups named `"infer"`:

```python
# This cannot be validated inside DataRequestConfig alone since it doesn't
# know its group name. Instead, add a validator on DataRequestDefinition.
@model_validator(mode="after")
def reject_augment_on_infer(self) -> DataRequestDefinition:
    for group_name, group_value in self.root.items():
        if group_name == "infer":
            for friendly_name, config in group_value.items():
                if config.augment:
                    raise ValueError(
                        f"Augmentation cannot be enabled on 'infer' data group "
                        f"(dataset '{friendly_name}'). ..."
                    )
    return self
```

**Why on `DataRequestDefinition`:** The per-dataset `DataRequestConfig` doesn't know which group it belongs to, so the "infer" guard must be at the definition level.

**Tests:** `tests/hyrax/test_data_request_config.py`
- `augment=True` on `train` group passes validation.
- `augment=True` on `infer` group raises `ValueError`.
- `augment` defaults to `None` (absent).
- Round-trip through `as_dict` preserves the field.

---

## Step 2: HyraxDataset — Add `on_epoch_start`

**File:** `src/hyrax/datasets/dataset_registry.py`

Add one method to `HyraxDataset`:

```python
def on_epoch_start(self):
    """Called at the beginning of each training epoch.
    Override in subclasses for epoch-level state resets (e.g. augmentation bookkeeping)."""
    pass
```

This is a plain method with a default no-op implementation — no registration or decorator needed.

Note: `row_cache_key` from the spec is **deferred to a future version**. V1 does not add a dataset-level cache key interface. Instead, the existing DataCache caches `get_<field>` results as it does today, and `augment_<field>` is applied post-cache (see Step 3c and Step 4).

**Tests:** `tests/hyrax/test_augmentation.py`
- `on_epoch_start()` is callable (no-op on base class).

---

## Step 3: DataProvider — Augmentation dispatch in `resolve_data`

**File:** `src/hyrax/datasets/data_provider.py`

### 3a: Store augmentation config and augment getters during `prepare_datasets`

In `prepare_datasets()`, after populating `self.dataset_getters`, also populate:

```python
self.augment_getters = {}  # friendly_name -> {field_name: augment_func}
self.augment_enabled = {}  # friendly_name -> bool
```

For each friendly_name where `augment` is truthy in the data_request entry:
- Set `self.augment_enabled[friendly_name] = True`
- Scan `dir(dataset_instance)` for methods starting with `augment_` and store them in `self.augment_getters[friendly_name]`.

### 3b: RNG seed generation

Add RNG infrastructure attributes in `__init__`:

```python
self._augment_master_seed = config.get("seed", None) or config["data_set"].get("seed", None)
self._current_epoch = 0
```

Add a helper to generate per-row rng seeds deterministically from the master seed, epoch, and index:

```python
def _augment_rng_seed(self, idx: int) -> np.int64:
    """Generate a deterministic, epoch-varying rng_seed for augmentation."""
    import hashlib
    seed_bytes = hashlib.sha256(
        f"{self._augment_master_seed}:{self._current_epoch}:{idx}".encode()
    ).digest()[:8]
    return np.int64(int.from_bytes(seed_bytes, "little"))
```

The rng_seed **varies per epoch** so that augmented data differs across training epochs (e.g. different random rotations each epoch). `self._current_epoch` starts at 0 and is incremented by `on_epoch_start` (Step 3d).

The use of a hash ensures:
- Determinism from the master seed (reproducible runs)
- No collisions between nearby indices
- Different augmentations each epoch for the same index

### 3c: Modify `resolve_data`

#### Critical design constraint: augmentation is a post-cache layer

**Do NOT skip the cache when augmentation is enabled.** The whole point of this
design is that `get_<field>` results (the expensive part — FITS I/O, network
reads, etc.) remain cached in DataCache exactly as they are today. Augmentation
functions are cheap transforms (rotations, flips) that run on top of cached base
data every time `resolve_data` is called.

The current code has an early-return on cache hit:

```python
cached_data = self.data_cache.try_fetch(idx)
if cached_data is not None:
    return cached_data          # <-- today this is the final return
```

**This early return must be removed when augmentation is active.** A cache hit
means the `get_<field>` data is ready — but augmentation still needs to run.
The wrong instinct is to bypass the cache entirely for augmented rows; that would
force expensive I/O on every access and defeat the purpose.

#### Concrete before/after

**Current flow** (simplified from `resolve_data` lines 806-855):
```python
def resolve_data(self, idx):
    # 1. Try cache
    cached = self.data_cache.try_fetch(idx)
    if cached is not None:
        return cached                        # early return — done

    # 2. Cache miss: call get_<field> for every field
    returned_data = {}
    for friendly_name, fields in self.requested_fields.items():
        getters = self.dataset_getters[friendly_name]
        data_dict = {field: getters[field](idx) for field in fields}
        returned_data[friendly_name] = data_dict

    # 3. Store in cache and return
    self.data_cache.insert_into_cache(idx, returned_data)
    return returned_data
```

**New flow:**
```python
def resolve_data(self, idx):
    # 1. Try cache — same as before, but NO early return when augmenting
    cached = self.data_cache.try_fetch(idx)
    if cached is not None:
        base_data = cached
    else:
        # 2. Cache miss: call get_<field> for every field (UNCHANGED)
        base_data = {}
        for friendly_name, fields in self.requested_fields.items():
            getters = self.dataset_getters[friendly_name]
            data_dict = {field: getters[field](idx) for field in fields}
            base_data[friendly_name] = data_dict

        # 3. Store get_<field> results in cache (UNCHANGED)
        self.data_cache.insert_into_cache(idx, base_data)

    # 4. If no augmentation is active, return base_data directly (same as today)
    if not self._has_any_augmentation:
        return base_data

    # 5. Augmentation pass: deepcopy so we don't mutate the cache, then augment
    rng_seed = self._augment_rng_seed(idx)
    augmented_data = copy.deepcopy(base_data)
    for friendly_name, fields_data in augmented_data.items():
        if friendly_name == "object_id":
            continue
        if not self.augment_enabled.get(friendly_name, False):
            continue
        if fields_data is None:   # left outer join miss
            continue
        for field, value in fields_data.items():
            augment_fn = self.augment_getters.get(friendly_name, {}).get(field)
            if augment_fn is not None:
                augmented_data[friendly_name][field] = augment_fn(value, idx, rng_seed)
    return augmented_data
```

#### Walk-through: what happens on epoch 2, index 7, with caching on

1. `data_cache.try_fetch(7)` → **cache hit** (was populated during epoch 1).
2. `base_data` = cached `get_<field>` results (e.g. raw FITS pixel data). No I/O.
3. `_has_any_augmentation` is `True` → enter augment pass.
4. `rng_seed = hash(master_seed, epoch=2, idx=7)` — different from epoch 1.
5. `deepcopy(base_data)` — so the cache entry stays clean.
6. For each augment-enabled field, call `augment_<field>(cached_value, 7, rng_seed)`.
7. Return augmented data. The cache still holds the original `get_<field>` results.

The same index on epoch 3 would hit the cache again at step 1, get a different
`rng_seed` at step 4, and produce a different augmentation at step 6 — all
without any I/O.

#### Key properties

- **DataCache is unchanged** — it stores `get_<field>` results keyed by idx, exactly as today. See Step 4.
- **Cache hits benefit augmented rows** — expensive I/O is cached; only cheap augmentation reruns.
- When `augment` is off (or absent), the code path is identical to today — the `_has_any_augmentation` check short-circuits to `return base_data`, no deepcopy.
- When `augment` is on, `augment_<field>` is called for fields that have it; fields without fall back to their `get_<field>` value (V1 per spec).
- The same `rng_seed` is passed to all `augment_<field>` calls within a single row, enabling correlated augmentation across fields (e.g. same rotation for image and mask).
- `deepcopy` prevents augmentation from mutating the cached base data.
- `_has_any_augmentation` is a bool set once during `prepare_datasets` (true if any friendly_name has `augment=True`), avoiding per-call dictionary scans when augmentation is not in use.

### 3d: Add `on_epoch_start` to DataProvider

```python
def on_epoch_start(self):
    """Dispatch on_epoch_start to all active dataset instances."""
    self._current_epoch += 1
    for dataset in self.prepped_datasets.values():
        dataset.on_epoch_start()
```

**Tests:** `tests/hyrax/test_augmentation.py` (new file)

Use the established test pattern: `HyraxLoopback` model + `HyraxRandomDataset` (or a small subclass of it), wired through a `hyrax.Hyrax()` instance as seen in `conftest.py:loopback_hyrax`. `HyraxLoopback` is a no-op model that returns its input, making it ideal for verifying data flow without model interference.

Create a small test dataset that subclasses `HyraxRandomDataset` and adds `augment_image`:

```python
class AugmentedRandomDataset(HyraxRandomDataset):
    def augment_image(self, data, idx, rng_seed):
        # Simple deterministic augment: flip sign
        return -data
```

Test cases:
- With `augment=True`, `resolve_data` calls `augment_image` but not `augment_label` (no such method → fallback to `get_label`).
- With `augment=False` (or absent), only `get_*` is called — augment methods ignored.
- `rng_seed` is the same for all `augment_<field>` calls within the same row (correlation test).
- `rng_seed` differs between different indices.
- `on_epoch_start` dispatches to all dataset instances.
- With `use_cache=True`, `get_<field>` results are cached; augmented output is derived from cached base data.
- Without augmentation, caching behavior is identical to today.

---

## Step 4: DataCache — Do NOT modify this file

**File:** `src/hyrax/datasets/data_cache.py` — **zero modifications. Do not touch this file.**

DataCache already does exactly what augmentation needs: it caches `get_<field>`
results keyed by DataProvider index. Augmentation does not change what gets cached
or how the cache is keyed. All augmentation logic lives in `resolve_data` (Step 3c),
which calls DataCache's existing `try_fetch` and `insert_into_cache` methods
without any changes to their signatures or behavior.

**Why no changes are needed — the cache stores pre-augmentation data:**

| Scenario | What DataCache stores | What `resolve_data` returns |
|----------|----------------------|---------------------------|
| `augment=False` (or absent) | `get_<field>` results (same as today) | cached data directly (same as today) |
| `augment=True` | `get_<field>` results (same as today) | `deepcopy(cached) → augment_<field>` applied |

In both cases, DataCache stores and serves the same thing: raw `get_<field>`
output. The augmentation layer in `resolve_data` consumes DataCache output and
transforms it — DataCache is not aware augmentation exists.

The `row_cache_key` interface mentioned in `specs/augmentation.md` is **deferred
to a future version**. V1 does not need it because the existing cache already
provides the performance benefit (cached I/O), and augmented results do not need
to be cached (they are cheap to recompute).

**If you find yourself wanting to add conditional logic, new parameters, or
augmentation awareness to DataCache — stop. That means the `resolve_data` changes
in Step 3c are structured incorrectly. Go back and fix `resolve_data` instead.**

**Tests:**
- With `augment=False` and `use_cache=True`, caching works exactly as before.
- With `augment=True` and `use_cache=True`, `get_<field>` results are still cached. The augment functions run on every access but receive cached base data, so only the augmentation computation cost is repeated (not I/O).

---

## Step 5: Wire `on_epoch_start` into the training loop

**File:** `src/hyrax/pytorch_ignite.py`

In `create_trainer`, after creating the engine, add a handler:

```python
# This needs the DataProvider(s) to be accessible. We'll pass them via
# a new parameter or attach them to the engine.
```

**Problem:** `create_trainer` doesn't currently have access to the DataProvider. The trainer engine only receives the DataLoader, not the underlying DataProvider.

**Solution:** Add the `on_epoch_start` handler in `Train.run()` (in `src/hyrax/verbs/train.py`) where both `trainer` and `dataset` dict are in scope:

```python
from hyrax.pytorch_ignite import Events

@trainer.on(Events.EPOCH_STARTED)
def dispatch_epoch_start(engine):
    for provider in dataset.values():
        provider.on_epoch_start()
```

This is added in `Train.run()` between the `create_trainer()` call and the `trainer.run()` call (around line 203-233 in train.py).

**Tests:** (in `tests/hyrax/test_augmentation.py`)
- Integration test using the `loopback_hyrax` pattern (HyraxLoopback + AugmentedRandomDataset) with `augment=True` and `epochs=2`. Verify `on_epoch_start` is called on dataset instances at the start of each epoch by tracking call count in the test dataset subclass.

---

## Step 6: Default config

**File:** `src/hyrax/hyrax_default_config.toml`

No changes needed. The `augment` key lives in the `data_request` config (per friendly-name), not in `[data_set]`. Since Pydantic validation handles defaults (`None` / absent means no augmentation), and TOML configs only define what's explicitly set, no default config entry is required.

---

## File Change Summary

| File | Change |
|------|--------|
| `src/hyrax/config_schemas/data_request.py` | Add `augment` field to `DataRequestConfig`; add infer-group validation |
| `src/hyrax/datasets/dataset_registry.py` | Add `on_epoch_start()` to `HyraxDataset` |
| `src/hyrax/datasets/data_provider.py` | Add augment getter discovery, rng_seed generation, augment dispatch in `resolve_data`, `on_epoch_start` dispatch |
| `src/hyrax/datasets/data_cache.py` | No changes |
| `src/hyrax/verbs/train.py` | Wire `Events.EPOCH_STARTED` handler to call `DataProvider.on_epoch_start()` |
| `tests/hyrax/test_data_request_config.py` | Tests for `augment` config field and infer-group rejection |
| `tests/hyrax/test_augmentation.py` (new) | Tests for augment dispatch, rng_seed correlation, cache behavior, on_epoch_start |

---

## Implementation Order

1. **Step 2** — `HyraxDataset` base method (`on_epoch_start`). No dependencies.
2. **Step 1** — Config schema. No dependencies.
3. **Step 3** — DataProvider augmentation dispatch. Depends on Steps 1 and 2.
4. **Step 4** — No work (DataCache unchanged).
5. **Step 5** — Training loop wiring. Depends on Step 3.
6. Run `ruff check src/ tests/ && ruff format src/ tests/` and `pre-commit run --all-files`.
7. Run `python -m pytest -m "not slow"` to verify all tests pass.

---

## Resolved Design Decisions

1. **Epoch-varying rng_seed:** Yes. `rng_seed` incorporates `self._current_epoch` so augmented data varies across epochs. This is the right default since a near-future version will not have the same indexes every epoch anyway.

2. **Cache granularity / `row_cache_key`:** Deferred. V1 has no dataset-level cache key interface. The existing DataCache caches `get_<field>` results as today; augmentation is post-cache.

3. **`get_<field>` caching for augmented rows:** Yes — this is the whole point of the design. DataCache caches `get_<field>` results (the expensive I/O). `augment_<field>` runs post-cache on a deepcopy of the cached data, so augmentation benefits from cached base data without polluting the cache with augmented results.
