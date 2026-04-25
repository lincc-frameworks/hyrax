# HATS Dataset Interoperability Plan (LSDB ↔ Hyrax)

## Context

Hyrax dataset classes must provide:

1. `__init__(self, config, data_location=None)`
2. `__len__(self)`
3. `get_<field>(self, idx)` for each requested field
4. `get_<primary_id_field>(self, idx)` for the configured primary ID field

The current LSST-specific HATS path computes an entire catalog eagerly. We want a generic HATS dataset class that enables interoperability between LSDB and Hyrax while preserving Hyrax's `get_*` access model.

---

## Goals

1. Add a generic built-in dataset class: `HyraxHATSDataset`.
2. Support reading HATS catalogs through LSDB.
3. Provide dynamic `get_*` getters for HATS columns.
4. Keep API and behavior aligned with Hyrax dataset conventions.
5. Provide tests and docs for the MVP so work is durable if interrupted.

---

## Non-goals for Phase 1

1. Full lazy partition-aware row locator index.
2. Advanced cone/polygon searches at dataset initialization time.
3. Margin catalog join behavior.
4. Remote-object-store performance tuning.

These remain planned for subsequent phases.

---

## Target user experience

Users can configure:

```toml
[data_request.train.catalog]
dataset_class = "HyraxHATSDataset"
data_location = "/path/or/url/to/hats_catalog"
fields = ["object_id", "coord_ra", "coord_dec"]
primary_id_field = "object_id"
```

And Hyrax will call generated methods such as:

- `get_object_id(idx)`
- `get_coord_ra(idx)`
- `get_coord_dec(idx)`

---

## Architecture

### Class: `HyraxHATSDataset`

Location:

- `src/hyrax/datasets/hats_dataset.py`

Responsibilities:

1. Read HATS catalog via LSDB (`lsdb.read_hats`).
2. Materialize a pandas DataFrame (MVP implementation).
3. Dynamically add `get_<column>(idx)` methods for each column, preserving raw column names.
4. Expose `__len__` helper consistent with hyrax requirements for dataset.

### Column naming

Column names are preserved as-is from the HATS catalog — no normalization is applied.

`get_<column>` methods are generated using the raw column name as the suffix. For column names
that are valid Python identifiers (e.g., `object_id`, `coord_ra`), these methods can be called
via normal dot syntax:

```python
dataset.get_object_id(0)
dataset.get_coord_ra(1)
```

For column names that contain non-identifier characters (e.g., `mag-r`), use `getattr`:

```python
getattr(dataset, "get_mag-r")(2)
```

### Primary ID behavior

For phase 1, users should set `primary_id_field` to the exact column name present in the HATS catalog.

---

## Config surface (Phase 1)

### Required

- `data_location` (HATS root path/URL)

### Optional

- No phase-1 column-subsetting option is implemented in the dataset class itself.

Hyrax still controls which fields are *requested at runtime* via `data_request.<split>.<dataset>.fields`.

---

## Error handling

1. Missing `data_location` => `ValueError`
2. LSDB import/runtime issues bubble up naturally (explicit dependency)

---

## Testing strategy (Phase 1)

Create dedicated unit tests:

- `tests/hyrax/test_hats_dataset.py`

Tests:

1. Initialization and length with synthetic HATS catalog.
2. Dynamic getter generation and retrieval correctness.
3. Behavior when `fields` is set in data_request (dataset still exposes all column getters; DataProvider requests selected fields).

Use `lsdb.from_dataframe(...).write_catalog(tmp_path)` to create fixture data.

---

## Incremental phases after MVP

### Phase 2

1. Partition-aware lazy row access without full DataFrame compute.
2. Row bundle cache to avoid repeated row fetch.
3. Optional filter pushdown and spatial prefilter.

### Phase 3

1. Shared HATS backend utility for LSSTDataset + HyraxHATSDataset.
2. Interop helpers (`to_lsdb_catalog`).
3. Benchmarks and docs notebook.

---

## Deliverables checklist

- [x] Spec document in `specs/hats_dataset.md`
- [x] `HyraxHATSDataset` implementation
- [x] Export from `hyrax.datasets`
- [x] Unit tests
- [ ] Optional docs touch-ups
