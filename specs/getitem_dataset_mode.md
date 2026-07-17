# Getitem Dataset Mode

## Motivation

The current dataset interface requires authors to implement `get_<field>(self, idx)` methods — one per field. DataProvider discovers these methods by name, caches references, and calls them individually in `resolve_data`. This works well but can feel unnatural for dataset authors coming from PyTorch's `Dataset.__getitem__` convention, where a single method returns all data for an index.

This spec adds an alternative: dataset authors can implement `__getitem__(self, idx)` instead. The two modes are mutually exclusive — a dataset class must use one or the other.

## Design Overview

### Mode Detection

After constructing a dataset instance, DataProvider inspects the instance to determine its mode:

- **Has getitem**: `hasattr(dataset_instance, '__getitem__')` — `HyraxDataset` does not define `__getitem__`, so this is True only when the subclass (or metaprogramming in `__init__`) provides it.
- **Has getters**: `any(m.startswith("get_") for m in dir(dataset_instance))` — True when any `get_*` method exists on the instance.

If both are present, DataProvider raises a `RuntimeError`. This check is intentionally simple and instance-level, allowing dataset authors to use metaprogramming in `__init__` (e.g. dynamically creating `get_*` methods or dynamically assigning `__getitem__`).

**Note on metadata tables:** The deprecated metadata table feature in `HyraxDataset.__init__` auto-generates `get_*` methods on the instance. If a getitem-mode dataset also uses a metadata table, the conflict check will fire. This is intentional — combining the deprecated metadata table path with the new getitem mode is not a supported combination. This is treated as undefined behavior and not a design constraint.

### The `__getitem__` Contract

A getitem-mode dataset's `__getitem__(self, idx)` must return a flat dictionary mapping field names to values:

```python
def __getitem__(self, idx: int) -> dict[str, Any]:
    return {
        "image": self.images[idx],
        "label": self.labels[idx],
        "object_id": f"obj-{idx:05d}",
    }
```

This is the same shape as the `base_data` dict that DataProvider constructs from getter-mode datasets (i.e. `{field: getters[field](real_idx) for field in fields}`). DataProvider wraps it with friendly names and extracts the primary ID, just as it does for getter mode.

### Field Communication

After constructing the dataset instance and determining it is in getitem mode, DataProvider sets:

```python
dataset_instance.requested_fields = tuple(fields)
```

This tells the dataset which fields DataProvider is interested in. The dataset's `__getitem__` can use `self.requested_fields` to optimize what it computes and returns, but this is advisory — if the dataset returns extra fields, DataProvider filters to only the requested ones. If the dataset returns fewer fields than requested, DataProvider raises an error.

`HyraxDataset` will initialize `self.requested_fields = ()` as a default in its `__init__`, so the attribute always exists.

### Field Discovery

When the data_request config does not specify `fields` for a getitem-mode dataset, DataProvider needs to discover available fields. It does this by calling `dataset_instance[0]` and using the keys of the returned dict. This mirrors how getter-mode field discovery scans for `get_*` methods.

### Augmentation

Augmentation (`augment_<field>` methods) is not supported for getitem-mode datasets. If a data_request entry configures `augment` for a getitem-mode dataset, DataProvider raises a `RuntimeError` with a clear message.

Getitem-mode dataset authors who need augmentation should implement it inside their `__getitem__` method directly. A future version could add DataProvider-managed augmentation for getitem mode, but that is out of scope.

### Tracing

The trace system (`_setup_trace`) currently instruments individual `get_*` getters. For getitem-mode datasets, the per-field getter instrumentation is skipped. Instead, a single trace point can wrap the `__getitem__` call. The `collate_*` and dataset-level `collate` instrumentation remains unchanged since collation is orthogonal to data fetch mode.

## Changes

### Step 1: Add `requested_fields` to `HyraxDataset`

**File:** `src/hyrax/datasets/dataset_registry.py`

In `HyraxDataset.__init__`, initialize:

```python
self.requested_fields = ()
```

This ensures the attribute always exists on any dataset instance, regardless of mode. Getter-mode datasets will have it set but won't use it. Getitem-mode datasets can reference it in their `__getitem__`.

**Tests:** `tests/hyrax/test_data_provider.py`
- A getter-mode dataset instance has `requested_fields == ()` after construction.

### Step 2: Add mode detection to `DataProvider.prepare_datasets()`

**File:** `src/hyrax/datasets/data_provider.py`

After instantiating each dataset in `prepare_datasets()`, detect the mode:

```python
has_getitem = hasattr(dataset_instance, '__getitem__')
has_getters = any(m.startswith("get_") for m in dir(dataset_instance))

if has_getitem and has_getters:
    raise RuntimeError(
        f"Dataset '{dataset_class}' (friendly name '{friendly_name}') defines both "
        f"__getitem__ and get_* methods. A dataset must use one interface or the other. "
        f"Use get_<field>/augment_<field> methods OR __getitem__, not both."
    )
```

Store which datasets are in getitem mode:

```python
self._getitem_datasets: set[str] = set()  # friendly names using getitem mode
```

Add the friendly name to this set when getitem mode is detected.

For getitem-mode datasets, the rest of the `prepare_datasets` loop diverges:

1. **Field discovery**: If no `fields` specified, call `dataset_instance[0]` and use the returned dict's keys.
2. **Set `requested_fields`**: `dataset_instance.requested_fields = tuple(fields)`.
3. **Skip getter caching**: Do not populate `self.dataset_getters[friendly_name]` with per-field entries. Instead, store a reference to the dataset instance itself (DataProvider already has `self.prepped_datasets`).
4. **Collation discovery**: `collate_<field>` and `collate` methods are still discovered normally — collation is independent of fetch mode.
5. **Augmentation rejection**: If `augment_cfg` is truthy, raise a `RuntimeError`.

For getter-mode datasets, the existing logic is unchanged.

**Tests:** `tests/hyrax/test_data_provider.py`
- A dataset with both `__getitem__` and `get_*` raises `RuntimeError` with a clear message.
- A getitem-mode dataset is correctly detected (friendly name appears in `_getitem_datasets`).
- A getter-mode dataset is correctly detected (friendly name does not appear in `_getitem_datasets`).
- A getitem-mode dataset with `augment: true` in config raises `RuntimeError`.
- A getitem-mode dataset with no explicit `fields` in config discovers fields from `__getitem__(0)`.

### Step 3: Update `DataProvider.resolve_data()`

**File:** `src/hyrax/datasets/data_provider.py`

In the main loop over `self.requested_fields.items()`, branch on mode when constructing `base_data`:

```python
if friendly_name in self._getitem_datasets:
    raw = self.prepped_datasets[friendly_name][real_idx]
    base_data = {field: raw[field] for field in fields}
else:
    base_data = {field: getters[field](real_idx) for field in fields}
```

The rest of `resolve_data` (caching, primary ID extraction, timing) remains unchanged because `base_data` has the same shape in both modes.

**Primary ID extraction** for getitem-mode datasets works identically — the primary_id_field is just a field name, and its value appears in the returned dict. When `primary_id_field` is not in `fields`, DataProvider still needs to fetch it. For getitem mode, this means calling `__getitem__` and extracting just that field. To avoid a redundant `__getitem__` call, the existing code path at lines 951-958 of `data_provider.py` needs a small adjustment: when the primary dataset is in getitem mode and the primary_id_field is not in the requested fields, the object_id should be extracted from a `__getitem__` call. This can be handled by storing a getter for the primary_id_field even for getitem-mode datasets, or by calling `__getitem__` and extracting the field.

The simpler approach: for getitem-mode primary datasets, always include the `primary_id_field` in the internal requested fields list (even if the user's config doesn't list it). This matches the expectation that getitem-mode datasets return the primary ID in their dict.

**Tests:** `tests/hyrax/test_data_provider.py`
- `resolve_data` returns correct structure for a getitem-mode dataset: `{friendly_name: {field: value}, "object_id": "..."}`.
- `resolve_data` works with a getitem-mode primary dataset where `primary_id_field` is in `fields`.
- `resolve_data` works with a getitem-mode primary dataset where `primary_id_field` is NOT in `fields`.
- `resolve_data` correctly pairs getitem-mode with getter-mode datasets in a multimodal setup.
- Data caching works correctly for getitem-mode datasets (cache hit on second access).
- Join support works with a getitem-mode secondary dataset.

### Step 4: Update `_setup_trace()` for getitem mode

**File:** `src/hyrax/datasets/data_provider.py`

In `_setup_trace`, skip per-field getter instrumentation for getitem-mode datasets (since `self.dataset_getters[friendly_name]` won't have per-field entries). The dataset-level and field-level collate instrumentation remains unchanged.

**Tests:** `tests/hyrax/test_trace.py`
- Tracing doesn't crash when a getitem-mode dataset is present.

### Step 5: Clean up existing dataset `__getitem__` methods

**Files:**
- `src/hyrax/datasets/random/hyrax_random_dataset.py` — Remove `__getitem__` from `HyraxRandomDataset`. It is not used by DataProvider (DataProvider uses `get_*` methods). Also remove `Dataset` from its base classes since it was only there for PyTorch compatibility via `__getitem__`.
- `src/hyrax/datasets/hyrax_csv_dataset.py` — Remove the dummy `__getitem__` that returns `{}`.

These changes ensure existing datasets pass the mutual-exclusion check.

**Tests:**
- Existing tests for `HyraxRandomDataset` and `HyraxCSVDataset` continue to pass.
- `HyraxRandomDataset` no longer has `__getitem__`.

### Step 6: Update documentation

**Files to update:**

1. `docs/dataset_class_reference.rst` — Add a new section "Alternative: `__getitem__` mode" after the existing "Method-by-method guide" section. Document:
   - When to use getitem mode vs getter mode
   - The `__getitem__` contract (return a flat dict of `{field: value}`)
   - `self.requested_fields` — how to use it, that it's advisory
   - That augmentation is not supported in getitem mode
   - A complete minimal class example using `__getitem__`

2. `docs/data_flow.rst` — Update the "Data Format Summary" table to note that `dataset.__getitem__(idx)` is an alternative to `dataset.get_*(idx)`, producing the same per-item format.

3. `docs/pre_executed/external_dataset_class.ipynb` — Add a new section (Step 5 or similar) showing the same dataset rewritten using `__getitem__` mode. This gives users a side-by-side comparison and demonstrates when the simpler interface is appropriate.

### Step 7: Add an end-to-end test with a getitem-mode dataset

**File:** `tests/hyrax/test_data_provider.py`

Create a minimal getitem-mode test dataset:

```python
class _GetitemDataset(HyraxDataset):
    def __init__(self, config, data_location=None):
        rng = np.random.default_rng(42)
        self._images = rng.random((10, 3, 8, 8), dtype=np.float32)
        self._ids = [f"id-{i}" for i in range(10)]
        super().__init__(config)

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return {
            "image": self._images[idx],
            "object_id": self._ids[idx],
        }
```

Test cases using this dataset:
- Basic `resolve_data` returns correct values.
- `DataProvider.__len__` works.
- `DataProvider.ids()` works.
- `DataProvider.collate()` works.
- `sample_data()` works.
- Multimodal: one getitem-mode dataset + one getter-mode dataset in the same DataProvider.
- Field filtering: dataset returns extra fields, DataProvider returns only requested ones.
- `requested_fields` is set correctly on the instance.

### Step 8: Add a conftest fixture for getitem-mode DataProvider

**File:** `tests/hyrax/conftest.py`

Add a fixture parallel to the existing `data_provider` fixture that uses a getitem-mode dataset. This supports the tests in Step 7.

## Out of Scope

- DataProvider-managed augmentation for getitem-mode datasets. Authors handle augmentation inside `__getitem__` directly.
- Getitem mode for joined secondary datasets returning `None` on missing keys — the left outer join `None` handling in `resolve_data` works the same regardless of mode, since it's determined before the per-dataset data fetch.
- Changes to the `DataRequestConfig` pydantic schema — no new config keys are needed. The existing `fields`, `primary_id_field`, `augment`, etc. all apply to both modes (with `augment` rejected for getitem mode at runtime).

## Migration Notes

- `HyraxRandomDataset` loses its `__getitem__` method. Any code that called `dataset_instance[idx]` directly on a `HyraxRandomDataset` (outside of DataProvider) will break. The recommended path is to use DataProvider for all data access.
- `HyraxCSVDataset` loses its dummy `__getitem__`. This method returned `{}` and was not useful.
- External dataset classes that define both `__getitem__` and `get_*` methods will get a clear error message telling them to choose one mode.

## Implementation Status

### Completed

**Step 1 — `requested_fields` on HyraxDataset:** Done. `self.requested_fields = ()` added to `HyraxDataset.__init__` in `dataset_registry.py`.

**Step 2 — Mode detection in `prepare_datasets()`:** Done. After instantiation, DataProvider checks `hasattr(dataset_instance, '__getitem__')` and `any(m.startswith("get_") for m in dir(dataset_instance))`. Mutual exclusion raises `RuntimeError`. Two helper methods `_prepare_getitem_dataset` and `_prepare_getter_dataset` extracted from the original monolithic loop. `self._getitem_datasets: set[str]` tracks friendly names in getitem mode.

**Step 3 — `resolve_data()` branching:** Done. Three sites updated:
- Pre-fetch primary object ID for joins (getitem mode calls `dataset[idx]` and extracts the field)
- `base_data` construction (getitem mode calls `dataset[real_idx]` and filters to requested fields)
- Object ID fallback when `primary_id_field` not in requested fields

Also updated `get_object_id()` to handle getitem-mode primary datasets.

**Step 4 — Trace:** No code change needed. `_setup_trace` iterates `self.dataset_getters[friendly_name].items()` which is `{}` for getitem mode, so per-field instrumentation is naturally skipped.

**Step 5 (partial) — Remove `__getitem__` from existing datasets:**
- `HyraxRandomDataset`: `__getitem__` removed, `Dataset` base class removed, `torch.utils.data.Dataset` import removed.
- `HyraxCSVDataset`: dummy `__getitem__` (returned `{}`) removed.
- `FitsImageDataset`: `__getitem__` removed, `Dataset` base class removed, torch import removed. Ruff auto-fixed the file after.
- `LSSTDataset`: `__getitem__` removed, `Dataset` base class removed, torch import removed.
- `DownloadedLSSTDataset`: `__getitem__` removed (inherited LSSTDataset's removal).
- `InferenceDataset`: `__getitem__` **kept** — it is used directly outside DataProvider (by `lookup.py`, `reduce_dimensions.py`, `visualize.py`, and tests). `Dataset` base class removed; type hint on `InferenceDatasetWriter.__init__` changed from `Dataset` to `DataProvider | InferenceDataset`.
- `ResultDataset`: `__getitem__` **kept** — it is used directly outside DataProvider (by `convert_results.py`, tests). Never had `Dataset` base class.

**Steps 7-8 — Tests and fixtures:** Done. 17 new tests added to `test_data_provider.py` covering:
- Mode detection (getitem detected, getter not in set)
- Mutual exclusion error
- `resolve_data` structure, `__len__`, `ids()`, `sample_data()`, `collate()`
- Field discovery from `__getitem__(0)`
- Field filtering (extra fields ignored)
- `requested_fields` set on instance
- `primary_id_field` not in fields list
- Augmentation rejected
- Multimodal (getitem + getter together)
- Caching, different indices
- Default `requested_fields == ()` on getter-mode dataset

Fixture `getitem_data_provider` added to `conftest.py`.

### Remaining Work

**Fix `FitsImageDataset`, `LSSTDataset`, `DownloadedLSSTDataset`:** These datasets define both `__getitem__` and `get_*` methods, but their `__getitem__` is used directly in tests and production code (not through DataProvider). Need to restore `__getitem__` on these three. The mutual-exclusion check will fire if someone puts them in a `data_request`, which is fine — they're designed for DataProvider's getter mode. The same pattern as `ResultDataset` and `InferenceDataset` where `__getitem__` was kept.

Specifically:
- `src/hyrax/datasets/fits_image_dataset.py` — restore the removed `__getitem__`, restore `Dataset` base class and torch import (ruff already reformatted the file after our edit, so check current state carefully).
- `src/hyrax/datasets/lsst_dataset.py` — restore `__getitem__`, restore `Dataset` base class and torch import.
- `src/hyrax/datasets/downloaded_lsst_dataset.py` — restore `__getitem__`.
- Tests that use `dataset[idx]` on these classes: `test_downloaded_lsst_dataset.py` (lines 75, 118, 154, 201, 212).

After restoring, run `python -m pytest -m "not slow" -x -q` to verify.

**Step 6 — Documentation updates:** Not started. Three files to update:
1. `docs/dataset_class_reference.rst` — Add "Alternative: `__getitem__` mode" section.
2. `docs/data_flow.rst` — Update Data Format Summary table.
3. `docs/pre_executed/external_dataset_class.ipynb` — Add section showing getitem-mode rewrite.

**Pre-commit:** Ran once; ruff auto-fixed `fits_image_dataset.py` (removed unused import). Need to re-run after the remaining fixes to confirm clean. Doc builds may fail for unrelated reasons — only investigate if caused by our doc edits.
