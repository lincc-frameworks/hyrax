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
(analogous to a SQL left outer join) instead of requiring positional alignment.
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

- **Left outer join**: all primary items are preserved regardless of whether
  they have a match in joined secondary datasets.
- When a primary item has no match in a joined secondary, `resolve_data`
  returns `None` for that secondary's friendly name.
- The effective dataset length is always the primary dataset length.
- Joined items appear in the **same order as the primary dataset**.

### Collation with missing data

During `collate`, `None` entries from unmatched secondaries are handled as
follows:

- Matched items are aggregated normally into per-field lists.
- Unmatched items (`None`) are skipped during aggregation (not appended to
  field lists).
- A boolean mask `<friendly_name>__matched` is added to the collated batch
  dict for any secondary that has at least one unmatched item. The mask has
  length equal to the batch size, with `True` for matched items and `False`
  for unmatched ones.
- Downstream consumers use the `__matched` mask to distinguish real data
  (including real NaN values) from join misses.

### Index structures

At `DataProvider.__init__` time, after all datasets are instantiated:

1. **Reverse-index maps** are built for each joined secondary:
   `{str(key_value): int(secondary_index)}`. This is a single pass over the
   secondary dataset — O(N) time and O(N) memory.

2. At runtime, `resolve_data(idx)` translates:
   - `idx` → `object_id` via the primary dataset's ID getter
   - `object_id` → `secondary_idx` via the reverse-index map (O(1) dict
     lookup)
   - If the lookup returns `None`, the secondary has no match for this item.

### Multithreaded index building

Reverse maps for independent secondaries are built in parallel using
`concurrent.futures.ThreadPoolExecutor`. Each secondary's map is independent,
so there are no synchronization concerns. This reduces init time when
multiple large secondary datasets are joined.

### Persistent index cache

Reverse maps are serialized to disk as pickle files alongside the dataset's
`data_location`. A content-based fingerprint (SHA-256 of dataset length +
sampled keys) detects staleness and triggers a rebuild. This avoids the O(N)
init cost on repeated runs with the same data.

Cache files follow the naming pattern `.hyrax_join_cache_<friendly_name>.pkl`
and are excluded from version control via `.gitignore`.

### Affected methods

| Method           | Without joins (unchanged) | With joins                                     |
|------------------|---------------------------|------------------------------------------------|
| `__len__`        | `len(primary_dataset)`    | `len(primary_dataset)` (unchanged)             |
| `__getitem__`    | passes `idx` to all       | translates per-dataset via join maps            |
| `resolve_data`   | same `idx` everywhere     | primary idx → object_id → per-secondary lookup |
| `get_object_id`  | direct primary lookup     | direct primary lookup (unchanged)              |
| `ids`            | iterates `range(len)`     | works via `get_object_id` (transparent)         |
| `metadata`       | passes `idxs` directly    | translates per-dataset; unmatched items omitted |
| `collate`        | unchanged                 | skips None entries, adds `__matched` mask       |

### Error handling

- **Duplicate keys in secondary**: the last occurrence wins; a warning is
  logged identifying both indices.
- **Missing `get_<join_field>` method**: raises `RuntimeError` at init.
- **Zero overlap**: allowed without error or warning. All primary items are
  preserved; every secondary entry will be `None`.

### Integration with split_fraction

The existing `split_fraction` machinery in `setup_dataset` /
`create_splits_from_fractions` operates on `range(len(provider))`. Since
`__len__` always returns the primary dataset length (regardless of joins),
split behavior is identical to the non-join case.

## Performance characteristics

| Phase       | Cost                    | Notes                                          |
|-------------|-------------------------|-------------------------------------------------|
| Init        | O(N) per joined dataset | Single pass to build reverse map                |
| Per-item    | O(1) dict lookup        | Same as a hash-map join; negligible vs I/O      |
| Memory      | ~50–80 bytes per key    | For 100M items: ~5–8 GB per joined secondary    |

### Optimization opportunities (not yet implemented)

1. **Bulk ID retrieval**: Datasets could optionally expose a
   `get_all_<field>()` method that returns all values at once (e.g., from a
   catalog column), avoiding per-item Python call overhead during map
   construction.

## Files modified

- `src/hyrax/config_schemas/data_request.py` — `join_field` on
  `DataRequestConfig`, mutual-exclusivity validator
- `src/hyrax/datasets/data_provider.py` — `_build_join_indices`,
  updated `resolve_data`, `__len__`, `get_object_id`, `metadata`,
  `_translate_metadata_indices`, `collate`
- `tests/hyrax/test_data_provider.py` — tests covering matching,
  left outer join semantics, collation with `__matched` mask,
  ordering, error cases, and backward compatibility
- `.gitignore` — exclude `.hyrax_join_cache_*.pkl` files
