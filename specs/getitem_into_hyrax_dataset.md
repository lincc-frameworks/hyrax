# Plan: Push `__getitem__` into HyraxDataset

## Context

The previous agent added a dual-mode system to DataProvider — one code path for `get_*` datasets, another for `__getitem__` datasets, with mutual-exclusion checks. This made DataProvider complex (two prepare methods, branching in resolve_data, mode detection).

The new approach: give `HyraxDataset` a **default `__getitem__`** that calls `get_*`/`augment_*` methods internally. DataProvider always calls `dataset[idx]` and gets a flat dict back. Datasets that want custom behavior override `__getitem__`. No mode detection, no errors for mixing — just Python method resolution.

## Starting Point

**Build on current commit** (5e01a3a7). The previous agent's changes to dataset files (removing old `__getitem__` methods, removing `torch.utils.data.Dataset` base classes) are changes we want. We'll rework DataProvider and DataCache in-place.

## Implementation

### Phase 1: Refactor DataCache to per-dataset

**File:** `src/hyrax/datasets/data_cache.py`

Remove the `friendly_name` dimension. Each dataset instance gets its own DataCache.

- Constructor: `DataCache(config, dataset_instance, augment_active: bool)`
- `_base_cache: dict[int, dict]` (was `dict[str, dict[int, dict]]`)
- `_augment_cache: dict[np.int64, dict]` (same flattening)
- All methods drop the `friendly_name` parameter: `try_fetch(real_idx, rng_seed)`, `insert_base(real_idx, data)`, `insert_augmented(real_idx, rng_seed, data)`
- `_datasets` becomes `_dataset` (single instance)

### Phase 2: Add default `__getitem__` to HyraxDataset

**File:** `src/hyrax/datasets/dataset_registry.py`

Add to `HyraxDataset.__init__` (after metadata table processing so dynamic getters exist):

```python
# Auto-discover get_* and augment_* methods
self._field_getters = {
    m[4:]: getattr(self, m)
    for m in dir(self)
    if m.startswith("get_") and callable(getattr(self, m))
}
self._augment_getters = {
    m[8:]: getattr(self, m)
    for m in dir(self)
    if m.startswith("augment_") and callable(getattr(self, m))
    and m != "augment_cache_key"
}

# Set by DataProvider before __getitem__ calls
self._rng_seed = None
self._augment_fields = []
self._cache = None
```

Add default `__getitem__`:

```python
def __getitem__(self, idx):
    fields = self.requested_fields or tuple(self._field_getters.keys())

    # Cache check
    if self._cache is not None:
        rng_seed = self._rng_seed if self._augment_fields else None
        cached, already_augmented = self._cache.try_fetch(idx, rng_seed)
        if cached is not None and (already_augmented or not self._augment_fields):
            return cached
        if cached is not None:
            # Base cached, need augmentation
            augmented = self._apply_augmentation(cached, idx)
            self._cache.insert_augmented(idx, self._rng_seed, augmented)
            return augmented

    # Fetch base data from get_* methods
    base_data = {field: self._field_getters[field](idx) for field in fields}

    # Cache base data
    if self._cache is not None:
        self._cache.insert_base(idx, base_data)

    # Apply augmentation if active
    if self._augment_fields and self._rng_seed is not None:
        augmented = self._apply_augmentation(base_data, idx)
        if self._cache is not None:
            self._cache.insert_augmented(idx, self._rng_seed, augmented)
        return augmented

    return base_data
```

Add `_apply_augmentation` helper (moved from DataProvider, simplified):

```python
def _apply_augmentation(self, base_data, idx):
    new_fields = {}
    for field, value in base_data.items():
        augment_fn = self._augment_getters.get(field) if field in self._augment_fields else None
        if augment_fn is not None and isinstance(value, np.ndarray):
            value = value.view()
            value.flags.writeable = False
        new_fields[field] = augment_fn(value, idx, self._rng_seed) if augment_fn else value
    return new_fields
```

**Key behavior (confirmed):** When `requested_fields` is empty (default), `__getitem__` returns ALL fields from all `get_*` methods. This ensures `dataset[idx]` works standalone (outside DataProvider) without configuration.

**Augmentation (confirmed):** Augmentation works universally — if a dataset defines `augment_*` methods and DataProvider configures augmentation, it works regardless of whether the dataset overrides `__getitem__`. Custom `__getitem__` datasets that want DataProvider-managed augmentation should call `super().__getitem__()`. No rejection of augmentation config based on mode.

### Phase 3: Simplify DataProvider

**File:** `src/hyrax/datasets/data_provider.py`

**Remove:**
- `self.dataset_getters` dict — getters now live on dataset instances
- `self._getitem_datasets` set — no mode detection
- `_prepare_getitem_dataset()` method
- `_prepare_getter_dataset()` method
- `self.augment_getters` dict — augmentation now handled by dataset
- `self.data_cache` — each dataset has its own cache
- `_apply_augmentation()` method

**Simplify `prepare_datasets()`** to a single path:

```python
for friendly_name, dataset_definition in self.data_request.items():
    # ... (instantiation unchanged) ...

    # Field discovery
    if not dataset_definition.get("fields", []):
        if dataset_instance._field_getters:
            dataset_definition["fields"] = list(dataset_instance._field_getters.keys())
        else:
            sample = dataset_instance[0]
            dataset_definition["fields"] = list(sample.keys())

    # Set requested_fields on dataset
    fields = list(dataset_definition.get("fields", []))

    # Primary dataset: ensure primary_id_field is in fields
    primary_id_field = dataset_definition.get("primary_id_field")
    if primary_id_field not in (None, False):
        self.primary_dataset = friendly_name
        self.primary_dataset_id_field_name = primary_id_field
        self.primary_data_location = dataset_definition.get("data_location", None)
        if primary_id_field not in fields:
            fields.append(primary_id_field)

    dataset_instance.requested_fields = tuple(fields)

    # Augmentation: configure on the dataset, not DataProvider
    augment_cfg = dataset_definition.get("augment")
    if augment_cfg:
        self._has_any_augmentation = True
        if augment_cfg is True:
            augment_cfg = [f for f in fields if f in dataset_instance._augment_getters]
        for field_name in augment_cfg:
            if field_name not in dataset_instance._augment_getters:
                raise RuntimeError(...)
        dataset_instance._augment_fields = augment_cfg
        self.augment_enabled[friendly_name] = augment_cfg

    # Initialize per-dataset cache
    dataset_instance._cache = DataCache(
        dataset_specific_config, dataset_instance, bool(augment_cfg)
    )

    # Collation discovery (unchanged from current code)
    # Join field tracking (unchanged)
    # Metadata field tracking (unchanged)

    self.requested_fields[friendly_name] = tuple(fields)
```

**Simplify `resolve_data()`:**

```python
def resolve_data(self, idx):
    rng_seed = self._augment_rng_seed() if self._has_any_augmentation else None

    # Pre-fetch primary object ID for joins
    if self._join_maps:
        primary = self.prepped_datasets[self.primary_dataset]
        id_getter = primary._field_getters.get(self.primary_dataset_id_field_name)
        object_id_str = str(id_getter(idx)) if id_getter else str(primary[idx][self.primary_dataset_id_field_name])
    else:
        object_id_str = None

    result = {}
    for friendly_name, fields in self.requested_fields.items():
        dataset = self.prepped_datasets[friendly_name]

        # Join mapping
        if friendly_name in self._join_maps:
            real_idx = self._join_maps[friendly_name].get(object_id_str)
            if real_idx is None:
                result[friendly_name] = None
                continue
        else:
            real_idx = idx

        # Set rng_seed on dataset instance
        dataset._rng_seed = rng_seed if self.augment_enabled.get(friendly_name) else None

        # Fetch — dataset handles caching and augmentation
        raw = dataset[real_idx]

        # Filter to requested fields
        result[friendly_name] = {field: raw[field] for field in fields}

    # Object ID extraction (unchanged logic, simpler code)
    ...

    return result
```

**Update `get_object_id()`** — use getter directly when available (avoids full `__getitem__` for just an ID):

```python
def get_object_id(self, idx):
    primary = self.prepped_datasets[self.primary_dataset]
    id_getter = primary._field_getters.get(self.primary_dataset_id_field_name)
    if id_getter:
        return str(id_getter(idx))
    return str(primary[idx][self.primary_dataset_id_field_name])
```

**Update `fields()`:**

```python
def fields(self):
    result = {}
    for friendly_name, dataset in self.prepped_datasets.items():
        if dataset._field_getters:
            result[friendly_name] = list(dataset._field_getters.keys())
        else:
            result[friendly_name] = list(self.requested_fields[friendly_name])
    return result
```

**Update `_setup_trace()`** — wrap getters on the dataset's `_field_getters` dict instead of DataProvider's `dataset_getters`:

```python
def _setup_trace(self):
    trace = get_trace()
    if trace is not None:
        trace.instrument_dataprovider(self)
        for friendly_name, dataset in self.prepped_datasets.items():
            for field_name, getter in dataset._field_getters.items():
                new_getter = trace.instrument_dataset_getter(dataset, getter, friendly_name, field_name)
                dataset._field_getters[field_name] = new_getter
            # collation instrumentation unchanged
```

**Update `_build_join_indices()`** — use `dataset._field_getters` instead of `self.dataset_getters`:

```python
# Validation:
getter = dataset._field_getters.get(join_field)
# In _build_one_map:
getter = secondary._field_getters[join_field]
```

### Phase 4: Update tests

**`tests/hyrax/test_data_provider.py`:**
- Remove references to `dp.dataset_getters` — use `dp.prepped_datasets[name]._field_getters` or remove those assertions
- Keep getitem-mode test classes (`_GetitemDataset`, `_ConflictDataset`) — `_ConflictDataset` no longer needed (no conflict error), remove it
- Update `_make_getitem_provider` and getitem tests to work with new design
- Remove `test_getitem_conflict_raises` (no conflict anymore)
- Remove `test_getitem_augment_rejected` — augmentation is now universally supported. Validation only rejects augmentation when the dataset has no matching `augment_*` method for the requested field (same validation as getter mode).

**`tests/hyrax/conftest.py`:**
- Update `getitem_data_provider` fixture for new internals
- Remove `data_cache` references if fixture creates one

**`tests/hyrax/test_data_cache.py`** (if exists):
- Update for per-dataset DataCache API (no friendly_name params)

**`tests/hyrax/test_downloaded_lsst_dataset.py`:**
- Update `dataset[idx]` expectations from `data_record["data"]["image"]` to `data_record["image"]` (flat dict from default `__getitem__`)

### Phase 5: Cleanup

- `specs/getitem_dataset_mode.md` — update Implementation Status to reflect the new approach
- Run `ruff check src/ tests/ && ruff format src/ tests/`
- Run `python -m pytest -m "not slow" -x -q`
- Run `pre-commit run --all-files`

## Files Modified

| File | Change |
|------|--------|
| `src/hyrax/datasets/data_cache.py` | Remove friendly_name dimension, per-dataset API |
| `src/hyrax/datasets/dataset_registry.py` | Add `_field_getters`, `_augment_getters`, default `__getitem__`, `_apply_augmentation` |
| `src/hyrax/datasets/data_provider.py` | Remove dual-mode code, simplify prepare/resolve, delegate to dataset |
| `tests/hyrax/test_data_provider.py` | Update for new internals, remove conflict test |
| `tests/hyrax/conftest.py` | Update fixtures |
| `tests/hyrax/test_downloaded_lsst_dataset.py` | Update `dataset[idx]` expectations to flat dict |
| `tests/hyrax/test_data_cache.py` | Update for per-dataset API (if this file exists) |

## What Stays Unchanged

- **Collation** — `DataProvider.collate()`, `custom_collate_functions`, `field_collate_functions` — batch assembly is orthogonal to fetch mode
- **PyTorch integration** — `pytorch_ignite.py` still passes DataProvider as a Dataset to DataLoader
- **InferenceDataset / ResultDataset** — keep their custom `__getitem__` overrides, work unchanged outside DataProvider
- **All existing `get_*` methods on datasets** — called by the default `__getitem__`, no changes needed
- **NaN handling** — stays in DataProvider.collate()

## Verification

1. `ruff check src/ tests/ && ruff format src/ tests/`
2. `python -m pytest -m "not slow" -x -q` — full fast test suite
3. `pre-commit run --all-files`
4. Manual check: create a simple `_GetitemDataset` (overrides `__getitem__`, no `get_*` methods) and verify DataProvider works with it alongside a getter-mode dataset

## Implementation Status

### Completed

**Phase 1 — DataCache refactored to per-dataset:** Done.
- `data_cache.py`: Constructor takes `(config, dataset, augment_active: bool)` instead of dicts.
- All methods drop the `friendly_name` parameter.
- `_base_cache` and `_augment_cache` are flat dicts keyed by `real_idx` / `aug_key`.

**Phase 2 — Default `__getitem__` on HyraxDataset:** Done.
- `dataset_registry.py`: `HyraxDataset.__init__` auto-discovers `_field_getters` and `_augment_getters`.
- New attributes: `_rng_seed`, `_augment_fields`, `_cache` (set by DataProvider).
- Default `__getitem__` calls `get_*` methods for `requested_fields` (or all fields when empty), handles caching and augmentation.
- `_apply_augmentation` helper moved from DataProvider into HyraxDataset.

**Phase 3 — DataProvider simplified:** Done.
- Removed: `dataset_getters`, `_getitem_datasets`, `_prepare_getitem_dataset`, `_prepare_getter_dataset`, `augment_getters`, `data_cache`, `_apply_augmentation`.
- `prepare_datasets()`: single unified path — field discovery from `_field_getters` or `__getitem__(0)`, augmentation configured on dataset instance, per-dataset DataCache initialized.
- `resolve_data()`: sets `_rng_seed` on dataset, calls `dataset[real_idx]`, filters to requested fields. No mode branching.
- `get_object_id()`: uses `_field_getters` directly when available for efficiency.
- `_setup_trace()`: wraps getters on `dataset._field_getters` instead of `self.dataset_getters`.
- `_build_join_indices()`: uses `dataset._field_getters` for join getters.
- `fields()`: queries `dataset._field_getters`.

**Phase 4 — Tests updated:** Done.
- `test_data_provider.py`: Removed `_ConflictDataset`, `test_getitem_conflict_raises`, `test_getitem_augment_rejected`. Updated references from `dp.dataset_getters` → `dp.prepped_datasets[name]._field_getters`. Updated `test_primary_id_field_fetched_when_not_in_fields` to reflect auto-inclusion of `primary_id_field`.
- `test_augmentation.py`: Updated DataCache construction calls to per-dataset API.
- `test_downloaded_lsst_dataset.py`: Changed `data_record["data"]["image"]` → `data_record["image"]` (flat dict from default `__getitem__`).

**Test results:** 514 passed, 0 failed (full fast suite `python -m pytest -m "not slow"`).

### Remaining Work

**Phase 5 — Cleanup:**
- Run `pre-commit run --all-files` and fix any issues.
- Update `specs/getitem_dataset_mode.md` to note it has been superseded by this plan.
- Step 6 from the original spec (documentation updates) is still not done:
  1. `docs/dataset_class_reference.rst` — add `__getitem__` mode section
  2. `docs/data_flow.rst` — update Data Format Summary table
  3. `docs/pre_executed/external_dataset_class.ipynb` — add getitem-mode example
