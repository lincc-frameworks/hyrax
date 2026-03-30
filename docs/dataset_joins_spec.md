# Dataset Joins — Design Specification

## Problem

Hyrax supports multimodal training and inference by retrieving data from
multiple datasets via `DataProvider`. Previously, all datasets in a group had
to be **index-aligned**: given an integer index `i`, every dataset had to
return data for the same real-world object at position `i`. This meant
datasets had to be pre-sorted or pre-filtered to share identical ordering —
a brittle requirement that breaks when datasets come from different sources
or pipelines.

## Goal

Allow datasets within a `DataProvider` group to be joined by a shared key
(analogous to a SQL inner join) instead of requiring positional alignment.
The implementation must:

1. **Not introduce substantial runtime overhead** — per-item access must stay
   O(1).
2. **Be pure Python** — no long-running services (SQL databases, etc.).
3. **Not require separate preprocessing steps** — index structures are built
   or loaded transparently when training or predicting begins.
4. **Be fully backward-compatible** — existing configs with index-aligned
   datasets continue to work without changes.

## Design

### Configuration

A new optional field, `join_field`, is added to `DataRequestConfig`:

```toml
[data_request.train.images]
dataset_class = "FitsImageDataset"
data_location = "/data/images"
primary_id_field = "object_id"      # designates this as the primary dataset
fields = ["image"]

[data_request.train.catalog]
dataset_class = "HyraxCSVDataset"
data_location = "/data/catalog.csv"
join_field = "object_id"            # join to primary by matching this field
fields = ["redshift", "magnitude"]
```

**Rules:**

- `join_field` and `primary_id_field` are **mutually exclusive** on the same
  dataset config. `join_field` is only for secondary datasets.
- Datasets **without** `join_field` remain index-aligned with the primary
  (existing behavior).
- `join_field` names a getter method (`get_<join_field>`) on the secondary
  dataset whose values match the primary dataset's `primary_id_field` values.

### Join semantics

- **Inner join**: only primary items whose key exists in **every** joined
  secondary dataset are included. Items missing from any secondary are
  dropped.
- The effective dataset length becomes the size of this intersection.
- Joined items appear in the **same relative order as the primary dataset**.

### Index structures

At `DataProvider.__init__` time, after all datasets are instantiated:

1. **Reverse-index maps** are built for each joined secondary:
   `{str(key_value): int(secondary_index)}`. This is a single pass over the
   secondary dataset — O(N) time and O(N) memory.

2. **Joined primary indices** are computed: the ordered list of primary
   indices whose keys appear in all secondary reverse maps. This is O(M)
   where M is the primary dataset length.

3. At runtime, `resolve_data(virtual_idx)` translates:
   - `virtual_idx` → `primary_idx` via `_joined_primary_indices[virtual_idx]`
   - `primary_idx` → `object_id` via the primary dataset's ID getter
   - `object_id` → `secondary_idx` via the reverse-index map (O(1) dict
     lookup)

### Affected methods

| Method           | Without joins (unchanged) | With joins                                   |
|------------------|---------------------------|----------------------------------------------|
| `__len__`        | `len(primary_dataset)`    | `len(_joined_primary_indices)`               |
| `__getitem__`    | passes `idx` to all       | translates per-dataset via join maps          |
| `resolve_data`   | same `idx` everywhere     | virtual → primary → per-secondary translation |
| `get_object_id`  | direct primary lookup     | translates virtual → primary first            |
| `ids`            | iterates `range(len)`     | works via `get_object_id` (transparent)       |
| `metadata`       | passes `idxs` directly    | translates per-dataset                        |
| `collate`        | unchanged                 | unchanged (operates on resolved dicts)        |

### Error handling

- **Empty intersection**: raises `RuntimeError` at init with a message
  identifying the primary and secondary datasets involved.
- **Duplicate keys in secondary**: the last occurrence wins; a warning is
  logged identifying both indices.
- **Missing `get_<join_field>` method**: raises `RuntimeError` at init.

### Integration with split_fraction

When joins are active, `__len__` returns the joined count. The existing
`split_fraction` machinery in `setup_dataset` / `create_splits_from_fractions`
operates on `range(len(provider))` — these are virtual indices that get
translated inside `resolve_data`. No changes to the split logic are needed.

## Performance characteristics

| Phase       | Cost                    | Notes                                          |
|-------------|-------------------------|-------------------------------------------------|
| Init        | O(N) per joined dataset | Single pass to build reverse map                |
| Per-item    | O(1) dict lookup        | Same as a hash-map join; negligible vs I/O      |
| Memory      | ~50–80 bytes per key    | For 100M items: ~5–8 GB per joined secondary    |

### Optimization opportunities (not yet implemented)

1. **Multithreaded index building**: For very large datasets (100M+ items),
   building reverse maps could be parallelized across secondary datasets
   using `concurrent.futures.ThreadPoolExecutor`. Each secondary's map is
   independent, so there are no synchronization concerns. The primary-index
   filtering step is sequential but fast (single pass with dict lookups).
   This is worth implementing if profiling shows init time is a bottleneck.

2. **Persistent index cache**: Reverse maps could be serialized to disk
   (e.g., as a pickle or SQLite file alongside the dataset) and reloaded on
   subsequent runs. A content-based fingerprint (e.g., hash of dataset
   length + first/last/sampled keys + file modification time) would detect
   staleness and trigger a rebuild. This avoids the O(N) init cost on
   repeated runs with the same data.

3. **Bulk ID retrieval**: Datasets could optionally expose a
   `get_all_<field>()` method that returns all values at once (e.g., from a
   catalog column), avoiding per-item Python call overhead during map
   construction.

## Files modified

- `src/hyrax/config_schemas/data_request.py` — `join_field` on
  `DataRequestConfig`, mutual-exclusivity validator
- `src/hyrax/datasets/data_provider.py` — `_build_join_indices`,
  updated `resolve_data`, `__len__`, `get_object_id`, `metadata`,
  `_translate_metadata_indices`
- `tests/hyrax/test_data_provider.py` — 8 tests covering matching,
  filtering, ordering, error cases, collation, and backward compatibility
