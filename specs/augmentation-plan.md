# Augmentation Implementation Plan

## Summary

Implement the augmentation system described in `specs/augmentation.md`. This adds `augment_<field_name>` methods to HyraxDataset classes, a new `augment` config key on data_request entries, modified data dispatch in DataProvider, and an `on_epoch_start` lifecycle callback. The existing DataCache continues to cache `get_<field>` results; `augment_<field>` is applied post-cache and its results are not cached in V1.

## Scope

**V1 only** — `augment` is a boolean. V2 (per-field dict) is out of scope but the design should not preclude it.

## V1 Implementation Status

V1 was implemented in commit `95e82028` (PR #942). All steps below are annotated with their implementation status and any divergences from the original plan.

---

## Step 1: Config Schema — Add `augment` field to `DataRequestConfig`

**Status:** ✅ Implemented as planned.

**File:** `src/hyrax/config_schemas/data_request.py`

Add to `DataRequestConfig`:

```python
augment: bool | list[str] | None = Field(
    None,
    description="Enable augmentation for this dataset. When true, augment_<field> methods are used. When a list, only the listed fields are augmented.",
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

**Status:** ✅ Implemented with divergence — signature includes `verb` parameter.

**File:** `src/hyrax/datasets/dataset_registry.py`

Add one method to `HyraxDataset`:

```python
def on_epoch_start(self, verb: str):
    """Called at the beginning of each epoch (or once for single-pass verbs).

    Parameters
    ----------
    verb : str
        Name of the verb that is running, e.g. ``"train"``, ``"infer"``,
        ``"test"``, or ``"engine"``.

    Override in subclasses to respond to epoch-level lifecycle events.
    """
    pass
```

This is a plain method with a default no-op implementation — no registration or decorator needed. The `verb` parameter was added during implementation to allow datasets to vary behavior by verb (e.g. skip augmentation bookkeeping during inference).

Note: `row_cache_key` from the spec is **deferred to a future version**. V1 does not add a dataset-level cache key interface. Instead, the existing DataCache caches `get_<field>` results as it does today, and `augment_<field>` is applied post-cache (see Step 3c and Step 4).

**Tests:** `tests/hyrax/test_augmentation.py`
- `on_epoch_start(verb)` is callable (no-op on base class).

---

## Step 3: DataProvider — Augmentation dispatch in `resolve_data`

**File:** `src/hyrax/datasets/data_provider.py`

**Status:** ✅ Implemented (see sub-steps for divergences).

### 3a: Store augmentation config and augment getters during `prepare_datasets`

**Status:** ✅ Implemented as planned.

In `prepare_datasets()`, after populating `self.dataset_getters`, also populate:

```python
self.augment_getters = {}  # friendly_name -> {field_name: augment_func}
self.augment_enabled = {}  # friendly_name -> bool
```

For each friendly_name where `augment` is truthy in the data_request entry:
- Set `self.augment_enabled[friendly_name] = True`
- Scan `dir(dataset_instance)` for methods starting with `augment_` and store them in `self.augment_getters[friendly_name]`.

### 3b: RNG seed generation

**Status:** ✅ Implemented with divergence — uses numpy RNG chain instead of hash.

Add RNG infrastructure attributes in `__init__`:

```python
# config["data_set"]["seed"] uses false as the Hyrax sentinel for "not set";
# treat it as None so numpy seeds from OS entropy rather than silently using 0.
_raw_seed = config["data_set"]["seed"]
_master_seed = None if _raw_seed is False else _raw_seed
self._augment_rng = np.random.default_rng(_master_seed)
self._epoch_rng = np.random.default_rng(int(self._augment_rng.integers(2**62)))
self._current_epoch = 0
```

Add a helper to draw the next seed from the epoch RNG:

```python
def _augment_rng_seed(self) -> np.int64:
    """Draw the next seed from the epoch RNG for one resolve_data call."""
    return self._epoch_rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, dtype=np.int64)
```

**Divergence from original plan:** The original plan used a SHA256 hash of `(master_seed, epoch, idx)` to produce per-index deterministic seeds. The implementation instead uses a two-level numpy RNG chain:
- `_augment_rng` is seeded once from the master seed and advances once per epoch to produce `_epoch_rng`.
- `_epoch_rng` is drawn from sequentially in `resolve_data` — one integer per call — so call order within an epoch determines the seed sequence.
- **Single-threaded access** is reproducible by call order (not by index).
- **Multi-threaded access** is intentionally non-reproducible (no locks on the hot path).

This approach was chosen because:
- It avoids the overhead of hashing on every `resolve_data` call.
- Per-index determinism is not needed since a near-future version will not have the same indexes every epoch anyway (balanced sampling will change the index sequence).
- The rng_seed still **varies per epoch** so augmented data differs across training epochs.

### 3c: Modify `resolve_data`

**Status:** ✅ Implemented with divergences — no deepcopy (uses new dict + read-only views), sequential RNG drawing instead of hash, tensorboard logging of augmentation_s, join-map index handling.

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

    # 5. Augmentation pass: build a new output dict so the cache is never mutated.
    #    Array references are reused for non-augmented fields; augment_<field> returns
    #    a new array for augmented ones, so no copies are needed here.
    rng_seed = self._augment_rng_seed()
    augmented_data = {}
    for friendly_name, fields_data in base_data.items():
        if friendly_name == "object_id":
            augmented_data[friendly_name] = fields_data
            continue
        if not self.augment_enabled.get(friendly_name, False):
            augmented_data[friendly_name] = fields_data
            continue
        if fields_data is None:   # left outer join miss
            augmented_data[friendly_name] = None
            continue

        # Use dataset-local indices for join-map secondaries.
        dataset_idx = idx
        if friendly_name in self._join_maps:
            dataset_idx = self._join_maps[friendly_name].get(base_data["object_id"])

        new_fields = {}
        for field, value in fields_data.items():
            augment_fn = self.augment_getters.get(friendly_name, {}).get(field)
            if augment_fn is not None and isinstance(value, np.ndarray):
                value = value.view()
                value.flags.writeable = False
            new_fields[field] = (
                augment_fn(value, dataset_idx, rng_seed) if augment_fn is not None else value
            )
        augmented_data[friendly_name] = new_fields
    return augmented_data
```

#### Walk-through: what happens on epoch 2, index 7, with caching on

1. `data_cache.try_fetch(7)` → **cache hit** (was populated during epoch 1).
2. `base_data` = cached `get_<field>` results (e.g. raw FITS pixel data). No I/O.
3. `_has_any_augmentation` is `True` → enter augment pass.
4. `rng_seed = self._epoch_rng.integers(...)` — drawn sequentially from the epoch RNG, which was reseeded at the start of epoch 2.
5. Build a new `augmented_data` dict. Non-augmented fields share references to cached arrays. For augmented ndarray fields, pass a read-only view to `augment_<field>`.
6. For each augment-enabled field, call `augment_<field>(read_only_view, 7, rng_seed)`.
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
- A new output dict is built instead of deepcopy — non-augmented fields share references; augmented ndarray fields get read-only views to prevent cache mutation. This avoids the cost of deepcopy.
- `_has_any_augmentation` is a bool set once during `prepare_datasets` (true if any friendly_name has `augment=True`), avoiding per-call dictionary scans when augmentation is not in use.
- Augmentation time is logged as `augmentation_s` to tensorboard.
- For secondary (joined) datasets, the dataset-local index from the join map is passed to `augment_<field>` instead of the DataProvider-level index.

### 3d: Add `on_epoch_start` to DataProvider

**Status:** ✅ Implemented with divergence — reseeds epoch RNG and accepts `verb` parameter.

```python
def on_epoch_start(self, verb: str):
    """Reset the epoch RNG and dispatch on_epoch_start to all dataset instances."""
    self._epoch_rng = np.random.default_rng(int(self._augment_rng.integers(2**62)))
    self._current_epoch += 1
    for dataset in self.prepped_datasets.values():
        dataset.on_epoch_start(verb)
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

**Status:** ✅ Confirmed — no changes made.

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
| `augment=True` | `get_<field>` results (same as today) | new dict with read-only views → `augment_<field>` applied |

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

## Step 5: Wire `on_epoch_start` into verbs

**Status:** ✅ Implemented with divergence — wired into all verbs, not just train.

### 5a: Training verb

**File:** `src/hyrax/verbs/train.py`

**Problem:** `create_trainer` doesn't currently have access to the DataProvider. The trainer engine only receives the DataLoader, not the underlying DataProvider.

**Solution:** Add the `on_epoch_start` handler in `Train.run()` where both `trainer` and `dataset` dict are in scope:

```python
from hyrax.pytorch_ignite import Events

@trainer.on(Events.EPOCH_STARTED)
def dispatch_epoch_start(engine):
    for provider in dataset.values():
        provider.on_epoch_start("train")
```

This is added in `Train.run()` between the `create_trainer()` call and the `trainer.run()` call.

### 5b: Single-pass verbs (infer, test, engine)

**Files:** `src/hyrax/verbs/infer.py`, `src/hyrax/verbs/test.py`, `src/hyrax/verbs/engine.py`

Single-pass verbs call `on_epoch_start` once before execution begins, passing the verb name:

- **infer.py:** Iterates all providers in the dataset dict and calls `provider.on_epoch_start("infer")`
- **test.py:** Iterates all providers and calls `provider.on_epoch_start("test")`
- **engine.py:** Calls `infer_dataset.on_epoch_start("engine")` on the infer provider

This ensures the epoch RNG is initialized for augmentation even in non-training contexts (e.g. Test Time Augmentation).

**Tests:** (in `tests/hyrax/test_augmentation.py`)
- Integration test using the `loopback_hyrax` pattern (HyraxLoopback + AugmentedRandomDataset) with `augment=True` and `epochs=2`. Verify `on_epoch_start` is called on dataset instances at the start of each epoch by tracking call count in the test dataset subclass.
- Epoch dispatch test covers two dataset instances to verify all providers are dispatched.

---

## Step 6: Default config

**Status:** ✅ Confirmed — no changes needed.

**File:** `src/hyrax/hyrax_default_config.toml`

No changes needed. The `augment` key lives in the `data_request` config (per friendly-name), not in `[data_set]`. Since Pydantic validation handles defaults (`None` / absent means no augmentation), and TOML configs only define what's explicitly set, no default config entry is required.

---

## File Change Summary

| File | Change | V1 Status |
|------|--------|-----------|
| `src/hyrax/config_schemas/data_request.py` | Add `augment` field to `DataRequestConfig`; add infer-group validation | ✅ Done |
| `src/hyrax/datasets/dataset_registry.py` | Add `on_epoch_start(verb)` to `HyraxDataset` | ✅ Done |
| `src/hyrax/datasets/data_provider.py` | Add augment getter discovery, rng_seed generation, augment dispatch in `resolve_data`, `on_epoch_start` dispatch | ✅ Done |
| `src/hyrax/datasets/data_cache.py` | No changes | ✅ Confirmed |
| `src/hyrax/verbs/train.py` | Wire `Events.EPOCH_STARTED` handler to call `DataProvider.on_epoch_start("train")` | ✅ Done |
| `src/hyrax/verbs/infer.py` | Call `on_epoch_start("infer")` before execution | ✅ Done (added during implementation) |
| `src/hyrax/verbs/test.py` | Call `on_epoch_start("test")` before execution | ✅ Done (added during implementation) |
| `src/hyrax/verbs/engine.py` | Call `on_epoch_start("engine")` before execution | ✅ Done (added during implementation) |
| `tests/hyrax/test_data_request_config.py` | Tests for `augment` config field and infer-group rejection | ✅ Done |
| `tests/hyrax/test_augmentation.py` (new) | Tests for augment dispatch, rng_seed correlation, cache behavior, on_epoch_start | ✅ Done |

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

1. **Epoch-varying rng_seed:** Yes. `rng_seed` varies per epoch via the two-level numpy RNG chain (`_augment_rng` → `_epoch_rng`), so augmented data differs across epochs. This is the right default since a near-future version will not have the same indexes every epoch anyway.

2. **Cache granularity / `row_cache_key`:** Deferred. V1 has no dataset-level cache key interface. The existing DataCache caches `get_<field>` results as today; augmentation is post-cache.

3. **`get_<field>` caching for augmented rows:** Yes — this is the whole point of the design. DataCache caches `get_<field>` results (the expensive I/O). `augment_<field>` runs post-cache on a new output dict with read-only ndarray views, so augmentation benefits from cached base data without polluting the cache with augmented results.

4. **RNG approach:** Sequential numpy RNG drawing (not per-index hash). Reproducible by call order in single-threaded mode; intentionally non-reproducible under multi-threading (no locks on the hot path). Per-index determinism was dropped because balanced sampling will change index sequences in a future version.

5. **Memory safety:** New output dict with shared references for non-augmented fields and read-only ndarray views for augmented fields, instead of `deepcopy`. Avoids the cost of deep-copying large arrays on every `resolve_data` call.

6. **`on_epoch_start` verb parameter:** The `verb: str` parameter was added to allow datasets to vary behavior by verb context. All verbs dispatch `on_epoch_start` — the `train` verb via `Events.EPOCH_STARTED`, and single-pass verbs (`infer`, `test`, `engine`) with a single call before execution.
