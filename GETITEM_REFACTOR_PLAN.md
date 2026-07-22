# Plan: HyraxDataset __getitem__ Refactor

## Context

The current dataset interface has DataProvider reaching into datasets via `get_*` method discovery, managing augmentation RNG, and calling individual getters per field. This creates a tangled relationship where DataProvider does too much and datasets are passive method bags.

This refactor:
1. Gives `HyraxDataset` a default `__getitem__` that calls `get_*` methods internally
2. Makes DataProvider delegate to `dataset[idx]` for all data fetching (golden path)
3. Moves augmentation RNG into the dataset (torch-based, epoch-varying)
4. Removes the deprecated metadata/metadata_table system entirely
5. Rewrites the visualize verb to use normal field access

**Simplicity bar**: Someone implementing just `__getitem__` + `__len__` on a HyraxDataset subclass should have it "just work" with no other requirements.

**Key constraints**:
- Join secondaries MUST have a `get_<join_field>` method (pure __getitem__ datasets cannot be join secondaries)
- Custom `__getitem__` + `augment` config → RuntimeError (augmentation only works with default __getitem__)
- DataCache is refactored to per-dataset (not replaced), living on HyraxDataset instance

---

## Step 1: Remove Metadata System — DONE

Remove with no backwards compatibility.

### `src/hyrax/datasets/dataset_registry.py`
- Remove `metadata_table` parameter from `HyraxDataset.__init__`
- Remove `self._metadata_table` storage
- Remove the auto-generated `get_*` methods from metadata table columns (lines 92-103)
- Remove `metadata_fields()` method
- Remove `metadata()` method
- Remove `numpy.ma` import inside metadata()

### `src/hyrax/datasets/data_provider.py`
- Remove `self.all_metadata_fields` from `__init__`
- Remove metadata field collection in `prepare_datasets()` (lines 797-803)
- Remove `metadata()` method (lines 1200-1272)
- Remove `metadata_fields()` method (lines 1274-1312)
- Remove `_translate_metadata_indices()` method (lines 1314-1339)

### `src/hyrax/datasets/random/hyrax_random_dataset.py`
- Remove metadata table construction in `HyraxRandomDatasetBase.__init__` (the `Table(meta)` creation and the `metadata_fields` config usage)
- Change `super().__init__(config, metadata_table, "object_id")` → `super().__init__(config)`
- Remove `from astropy.table import Table` import

### `src/hyrax/datasets/inference_dataset.py`
- Remove `metadata_fields()` override (lines 227-236)
- Remove `metadata()` override (lines 238-274)

### `src/hyrax/datasets/kafka_stream_dataset.py`
- Change `super().__init__(config, metadata_table=None)` → `super().__init__(config)`

### `src/hyrax/datasets/lsst_dataset.py`
- Change `super().__init__(config, self.catalog, self.oid_column_name)` → `super().__init__(config)`
- Add explicit `get_object_id(self, idx)` — previously auto-generated from metadata table

### `src/hyrax/datasets/fits_image_dataset.py`
- Remove `_prepare_metadata()` call from `__init__`, change super to `super().__init__(config)`
- Note: catalog metadata fields (ira, idec, SNR) are no longer auto-exposed as `get_*` methods

### `src/hyrax/datasets/hsc_dataset.py`
- Change `super().__init__(config, data_location)` → `super().__init__(config, data_location=data_location)` (was passing data_location positionally where metadata_table used to be)

### Config / tests
- Remove `metadata_fields` key from `[data_set.HyraxRandomDataset]` in `hyrax_default_config.toml`
- Update/remove tests that exercise metadata methods

---

## Step 2: Add Default `__getitem__` to HyraxDataset

### `src/hyrax/datasets/dataset_registry.py` — HyraxDataset.__init__

After storing `self._config`, discover and cache methods:

```python
self._field_getters: dict[str, Callable] = {}
self._augment_getters: dict[str, Callable] = {}

for name in dir(self):
    if name.startswith("get_") and callable(getattr(self, name, None)):
        self._field_getters[name[4:]] = getattr(self, name)
    elif name.startswith("augment_") and name != "augment_cache_key" and callable(getattr(self, name, None)):
        self._augment_getters[name[8:]] = getattr(self, name)

self.requested_fields: tuple[str, ...] = ()
self._epoch: int = 0
self._augment_enabled: bool = False  # Set by DataProvider
```

### `src/hyrax/datasets/dataset_registry.py` — HyraxDataset.__getitem__

```python
def __getitem__(self, idx: int) -> dict[str, Any]:
    fields = self.requested_fields or tuple(self._field_getters.keys())

    # Check base cache
    cached = self._data_cache.try_fetch_base(idx)
    if cached is not None and not self._augment_enabled:
        return cached

    # Fetch base data (from cache or getters)
    if cached is not None:
        base_data = cached
    else:
        base_data = {field: self._field_getters[field](idx) for field in fields}
        self._data_cache.insert_base(idx, base_data)

    if not self._augment_enabled:
        return base_data

    # Generate per-call RNG seed: same for all fields, varies by epoch
    import torch
    rng_seed = np.int64(hash((torch.initial_seed(), self._epoch, idx)) % (2**63 - 1))

    # Check augment cache
    aug_key = self.augment_cache_key(idx, rng_seed)
    if aug_key is not None:
        aug_cached = self._data_cache.try_fetch_augmented(aug_key)
        if aug_cached is not None:
            return aug_cached

    # Apply augmentation
    result = {}
    for field, value in base_data.items():
        augment_fn = self._augment_getters.get(field)
        if augment_fn is not None:
            if isinstance(value, np.ndarray):
                value = value.copy()
                value.flags.writeable = False
            result[field] = augment_fn(value, idx, rng_seed)
        else:
            result[field] = value

    if aug_key is not None:
        self._data_cache.insert_augmented(aug_key, result)

    return result
```

### `src/hyrax/datasets/dataset_registry.py` — on_epoch_start update

```python
def on_epoch_start(self, verb: str):
    self._epoch += 1
```

### Data cache integration

Each dataset instance owns its own `DataCache` (see Step 7 for the refactor of `data_cache.py`). Created in `HyraxDataset.__init__`:

```python
from hyrax.datasets.data_cache import DataCache
self._data_cache = DataCache(use_cache=config["data_set"]["use_cache"])
```

---

## Step 3: Simplify DataProvider

### Key files to read first
- `src/hyrax/datasets/data_provider.py` — the whole file, focus on `prepare_datasets()`, `resolve_data()`, `__init__`
- `src/hyrax/trace.py` — understand `instrument_dataprovider` and `instrument_dataset_getter`

### `prepare_datasets()` changes

After instantiating each dataset:

1. **Field discovery**: If no `fields` in config:
   - If dataset has `_field_getters` (getter-mode): use those keys
   - Else (custom `__getitem__`): call `dataset[0]` and use the returned dict's keys

2. **Set `dataset.requested_fields`**: Set it to the tuple of resolved field names.

3. **Augmentation setup**: If `augment` config is truthy:
   - If dataset has overridden `__getitem__` (custom mode): raise RuntimeError
   - Otherwise: set `dataset._augment_enabled = True`, validate augment methods exist

4. **Collation discovery**: Unchanged — `collate_<field>` and `collate` methods still discovered

5. **Remove `self.dataset_getters`** — no longer needed for the data path

6. **Keep join_field validation**: If `join_field` configured, verify `get_<join_field>` exists on dataset. Raise RuntimeError if not.

### `_build_join_indices()` changes

Access getters via `dataset._field_getters[join_field]` instead of `self.dataset_getters`.

### `resolve_data()` simplification

```python
def resolve_data(self, idx: int) -> dict:
    result = {}

    # Pre-fetch primary object ID for joins
    if self._join_maps:
        primary = self.prepped_datasets[self.primary_dataset]
        object_id_str = str(primary._field_getters[self.primary_dataset_id_field_name](idx))
    else:
        object_id_str = None

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

        # Delegate to dataset — cache + augmentation handled there
        raw = dataset[real_idx]
        result[friendly_name] = {field: raw[field] for field in fields}

    # Extract object_id
    if self.primary_dataset:
        primary_data = result.get(self.primary_dataset, {})
        if primary_data and self.primary_dataset_id_field_name in primary_data:
            object_id = primary_data[self.primary_dataset_id_field_name]
        elif object_id_str is not None:
            object_id = object_id_str
        else:
            primary = self.prepped_datasets[self.primary_dataset]
            object_id = str(primary[idx][self.primary_dataset_id_field_name])
        result["object_id"] = str(object_id)

    return result
```

### `get_object_id()` and `ids()` changes

```python
def get_object_id(self, idx: int) -> str:
    primary = self.prepped_datasets[self.primary_dataset]
    getter = primary._field_getters.get(self.primary_dataset_id_field_name)
    if getter:
        return str(getter(idx))
    return str(primary[idx][self.primary_dataset_id_field_name])
```

### `fields()` method update

Return from `self.requested_fields` dict plus any additional discovered fields from `dataset._field_getters`.

### `_setup_trace()` changes

Skip per-field getter instrumentation. Instrument `dataset.__getitem__` as a single trace point per dataset. Collation tracing unchanged.

### Remove from DataProvider
- `self.dataset_getters`
- `self.all_metadata_fields`
- `self.augment_getters`, `self.augment_enabled`
- `self._augment_rng`, `self._epoch_rng`, `self._current_epoch`
- `_augment_rng_seed()`, `_apply_augmentation()`
- `metadata()`, `metadata_fields()`, `_translate_metadata_indices()`
- `self.data_cache` — DataCache now lives per-dataset on the HyraxDataset instance

### `on_epoch_start()` simplification

```python
def on_epoch_start(self, verb: str):
    for dataset in self.prepped_datasets.values():
        dataset.on_epoch_start(verb)
```

---

## Step 4: Clean Up Existing Datasets

### Key files to read
- `src/hyrax/datasets/random/hyrax_random_dataset.py`
- `src/hyrax/datasets/hyrax_csv_dataset.py`
- `src/hyrax/datasets/inference_dataset.py`
- `src/hyrax/datasets/fits_image_dataset.py`
- `src/hyrax/datasets/lsst_dataset.py`

### Changes
- `HyraxRandomDataset`: Remove `__getitem__`, remove `Dataset` from bases
- `HyraxCSVDataset`: Remove dummy `__getitem__`
- `InferenceDataset`: Remove `Dataset` from bases, update `__getitem__` to return field dict
- `FitsImageDataset` / `LsstDataset`: Check for `__getitem__`/`Dataset` base, remove if present

---

## Step 5: Rewrite Visualize Verb — DONE

### Key files to read
- `src/hyrax/verbs/visualize.py` — full file
- How `setup_dataset()` in `src/hyrax/pytorch_ignite.py` creates DataProviders

### Pattern replacement

`metadata_fields()` → `data_provider.fields()` (returns available field names per dataset)

`metadata(indices, fields)` → iterate indices, call `data_provider[idx]`, extract fields:
```python
def _batch_fetch_fields(self, indices, fields):
    import numpy as np
    result = {field: [] for field in fields}
    for idx in indices:
        sample = self.metadata_provider[idx]
        primary_name = self.metadata_provider.primary_dataset
        data = sample.get(primary_name, {})
        for field in fields:
            result[field].append(data.get(field))
    return {field: np.array(values) for field, values in result.items()}
```

### Specific changes
- Replace `self.metadata_provider.metadata_fields()` with DataProvider field discovery
- Replace `self.metadata_provider.metadata(indices, fields)` with batch fetch pattern
- data_request config for visualize must include needed fields in `fields` list

---

## Step 6: Mode Detection and Error Handling

In `DataProvider.prepare_datasets()`, detect custom __getitem__:

```python
has_custom_getitem = type(dataset_instance).__getitem__ is not HyraxDataset.__getitem__
```

Conditions to check:
- Custom `__getitem__` + `augment` config → RuntimeError
- Join secondary without `get_<join_field>` → RuntimeError
- Custom `__getitem__` with no `fields` in config and no `get_*` methods → call `dataset[0]` for field discovery

---

## Step 7: Refactor `DataCache` to Per-Dataset

### Key file: `src/hyrax/datasets/data_cache.py`

Currently `DataCache.__init__` takes `(config, datasets: dict[str, HyraxDataset], augment_active: dict[str, bool])` — manages caches for multiple datasets keyed by friendly name.

### Changes
- Constructor becomes `DataCache(use_cache: bool)` — one instance per dataset
- Remove `datasets` and `augment_active` constructor args
- Remove friendly-name keying from `_base_cache` and `_augment_cache` — each is now just `dict[int, dict]` and `dict[Any, dict]`
- Split `try_fetch` into `try_fetch_base(idx)` and `try_fetch_augmented(key)`
- `insert_base(idx, data)` and `insert_augmented(key, data)` — no friendly_name param
- Keep size tracking and logging

---

## Step 8: Tests

### Update existing tests
- Remove all metadata-related test assertions
- Update DataProvider tests to verify new `dataset[idx]` → dict flow
- Update HyraxRandomDataset tests (no `__getitem__`, no metadata table)

### New tests
- Pure `__getitem__` dataset works with DataProvider
- Getter-mode dataset works through default `__getitem__`
- Augmentation through default `__getitem__` (seed consistency, epoch variation)
- Custom `__getitem__` + `augment` config raises RuntimeError
- Join secondary without getter raises RuntimeError
- Field discovery from `dataset[0]`
- Cache hit/miss through default `__getitem__`
- `requested_fields` set correctly

---

## Open Considerations

1. **`pull_up_primary_dataset_methods`**: Already excludes `_`-prefixed methods, so `_field_getters`, `_data_cache` etc. are safe.

2. **InferenceDataset `__getitem__`**: Currently returns torch tensors. Must change to dict. Breaks direct `InferenceDataset[idx]` usage outside DataProvider.

3. **Streaming data provider**: `StreamingDataProvider` has its own flow. Not affected.

4. **Trace system**: Per-field getter instrumentation goes away. Single `__getitem__` trace point per dataset, or accept reduced granularity.

---

## Verification

1. `ruff check src/ tests/ && ruff format src/ tests/`
2. `python -m pytest -m "not slow"` — full fast test suite
3. `pre-commit run --all-files`
4. Manual: pure `__getitem__` dataset works with DataProvider
5. Manual: `HyraxRandomDataset` works through default `__getitem__`
6. Verify augmentation seed consistency across fields, variation across epochs
