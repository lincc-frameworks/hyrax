# HATS Dataset Phase 2 Plan (Lazy, Partition-Aware Access)

## Scope and intent

This document defines **Phase 2** for `HyraxHATSDataset`: move from full eager materialization (`catalog.compute()`) to a **lazy, partition-aware row access model** that remains compatible with Hyrax's `get_<field>(idx)` interface.

Phase 2 is intentionally implementation-oriented so we can review architecture and sequencing before coding.

---

## Why Phase 2 is needed

The current Phase 1 dataset loads an entire LSDB catalog into a pandas DataFrame. This is simple, but does not scale for large HATS catalogs and defeats LSDB/HATS advantages:

- HATS is partitioned by sky region (HEALPix), and LSDB is designed to read only needed partitions.
- Random row reads can be expensive if each read triggers broad scans.
- A Hyrax access pattern that reads many nearby indices should exploit locality.

Phase 2 will preserve Hyrax ergonomics while reducing memory pressure and improving large-catalog behavior.

---

## Phase 2 goals

1. **Do not materialize full catalog by default** during dataset initialization.
2. Support `__len__` and `get_<field>(idx)` with deterministic global row indexing.
3. Introduce a **row-bundle cache** keyed by partition (and optional row-group window) to avoid repeated remote reads.
4. Allow optional **column projection** and basic **filter pushdown**.
5. Keep existing Hyrax config usage stable where possible.
6. Add tests/benchmarks/docs that make behavior and tradeoffs explicit.

---

## Explicit non-goals for this phase

- Rewriting Hyrax's data request execution model.
- Full spatial query DSL in Hyrax config.
- Distributed scheduler tuning (Dask cluster orchestration).
- Margin catalog joins and crossmatch orchestration.

---

## Proposed architecture

## 1) New internal components in `hats_dataset.py`

### A. `HATSPartitionIndex`

A lightweight mapping between global row indices and HATS partition-local offsets.

Responsibilities:

- Read partition metadata once.
- Build cumulative row counts per partition.
- Resolve global index `idx` -> `(partition_id, local_offset)` via binary search.
- Expose total row count for `__len__`.

Notes:

- Source of truth should be LSDB/HATS metadata artifacts when available (avoid scanning data leaves).
- If row-count metadata is missing/inconsistent, fail with clear message and fallback options.

### B. `HATSRowBundleCache`

A bounded in-memory cache that stores recently accessed row bundles.

Responsibilities:

- Cache unit: `(partition_id, selected_columns, bundle_start, bundle_size)`.
- Configurable `max_bundles` and `bundle_size`.
- LRU eviction.
- Optional metrics counters (hit/miss, bytes, load time).

### C. `HATSLazyAccessor`

Execution layer that fetches rows from LSDB/HATS using partition-aware reads.

Responsibilities:

- Use `HATSPartitionIndex` to locate partition/local offset.
- Fetch only required columns where possible.
- Load a small contiguous bundle around requested local row to increase cache hit probability.
- Return scalar/array values normalized to current dataset semantics.

## 2) `HyraxHATSDataset` responsibilities after refactor

- Initialize LSDB catalog lazily (`lsdb.read_hats` / equivalent lazy-open path).
- Build normalized column map for getter generation.
- Instantiate `HATSPartitionIndex` and `HATSRowBundleCache`.
- Generate dynamic getters that route to `HATSLazyAccessor.get_value(idx, field)`.
- Keep `sample_data` behavior (read first row lazily).

---

## Data access strategy

## A. Global index resolution

Use cumulative row counts per partition:

- `partition_row_counts = [n0, n1, ..., nk]`
- `prefix = cumulative_sum(partition_row_counts)`
- Resolve via `bisect_right(prefix, idx)`.

Complexities:

- Build: `O(P)` for `P` partitions.
- Lookup: `O(log P)` per access.

## B. Bundle reads

Given `(partition, local_offset)`:

- Compute `bundle_start = floor(local_offset / bundle_size) * bundle_size`.
- Read rows `[bundle_start : bundle_start + bundle_size)` from that partition.
- Cache bundle.
- Extract requested row.

Rationale:

- Maintains bounded memory.
- Improves locality if training/evaluation iterates near-adjacent indices.
- Avoids one-file-per-row penalties.

## C. Column projection

Column set for bundle reads:

- Always include `primary_id_field`.
- Include fields requested in data request if present.
- Permit override: `hats_columns = "all" | [..]` (new config option).

Default for phase 2: projected columns = union of required columns for active getters in request context, with safe fallback to all columns if projection path is unavailable.

## D. Optional filter pushdown

Add optional config section:

```toml
[data_request.train.catalog.hats]
filters = [
  ["coord_ra", ">=", 120.0],
  ["coord_ra", "<", 130.0],
]
```

Apply at catalog-open/read stage using LSDB-compatible filtering primitives where available.

If a filter is unsupported, fail fast with precise error and examples.

---

## Configuration additions (proposed)

All options optional with conservative defaults.

```toml
[data_request.<split>.<dataset>.hats]
lazy = true                     # default true in phase 2
bundle_size = 256              # rows per partition bundle
max_cached_bundles = 64        # LRU capacity
project_columns = "auto"       # auto | all | explicit list
filters = []                   # optional predicate list
strict_metadata = true         # fail if partition row counts unavailable
```

Backward compatibility:

- Existing phase-1 configs should continue to work.
- We may keep an escape hatch `lazy=false` for troubleshooting.

---

## Error handling and edge cases

1. **Missing metadata for row counts**
   - If `strict_metadata=true`: raise `ValueError` with actionable guidance.
   - Else: fallback strategy (documented as potentially expensive).

2. **Index out of range**
   - Raise `IndexError` consistently from central accessor.

3. **Field missing from projected columns**
   - Raise `KeyError` mentioning normalized/original column names.

4. **Column name normalization collisions**
   - Keep deterministic suffix policy (`_2`, `_3`, ...), and retain reverse map in object state.

5. **Remote I/O failures / transient read errors**
   - Surface original exception class with contextual message (partition id, columns).

---

## Testing plan

Add or update: `tests/hyrax/test_hats_dataset.py` and dedicated lazy-access tests.

## Unit tests

1. **Partition index construction** from synthetic multi-partition catalogs.
2. **Global index mapping** correctness across partition boundaries.
3. **Getter correctness** under lazy reads for scalar and array-like fields.
4. **Cache behavior**: deterministic hit/miss with controlled access sequence.
5. **Column projection**: ensure only required columns requested (via mock/spies).
6. **Normalization collisions**: generated getter names are unique and stable.

## Integration tests

1. End-to-end Hyrax prepare/train split with lazy HATS dataset.
2. `fields` in data_request interacts correctly with projected reads.
3. Optional filter path returns reduced row domain with correct indexing semantics.

## Performance regression checks

Small benchmark script (non-CI optional):

- Compare phase-1 eager vs phase-2 lazy memory and wall-clock for:
  - sequential access of N rows,
  - random access of N rows,
  - clustered random access.

Record median metrics and add to docs.

---

## Implementation sequence

### Milestone 1: Internal scaffolding

- Add internal index/cache/accessor classes.
- Wire dataset getters through accessor.
- Preserve old behavior behind compatibility switch.

### Milestone 2: Metadata-driven indexing

- Implement robust partition row-count extraction.
- Add strict/fallback modes.
- Validate length and index mapping.

### Milestone 3: Bundle cache + projection

- Implement bundle fetch/cache.
- Add projection policy (`auto`, `all`, explicit list).
- Add cache metrics hooks.

### Milestone 4: Filter pushdown (optional, guarded)

- Parse filter config schema.
- Apply LSDB-supported filters and validation.

### Milestone 5: Hardening

- Extend tests (unit + integration).
- Add benchmark notes and developer docs.
- Finalize error messages and migration notes.

---

## Acceptance criteria (Phase 2 done)

1. No full-catalog `.compute()` on initialization in default lazy path.
2. `__len__` and dynamic `get_<field>(idx)` pass full test suite.
3. Repeated local accesses show measurable cache benefit in benchmark.
4. Existing phase-1 configs run unchanged (or with clearly documented warnings).
5. Docs clearly describe lazy semantics, config knobs, and limitations.

---

## Resolutions before implementation

1. Prefer `lsdb.open_catalog` where available, and fall back to `lsdb.read_hats` when required by the installed LSDB version or catalog behavior.
2. Start with the simplest viable row-count metadata path (derive counts via partition-aware length operations), then iterate if catalog variants require richer metadata parsing.
3. Preserve globally stable row ordering within a fixed underlying HATS representation; if HATS format evolution changes physical ordering, index drift across versions is acceptable and should be documented.
4. Keep cache stats in debug logging only for Phase 2 (no public user-facing stats API yet).

---

## References

- Existing phase-1 spec: `specs/hats_dataset.md`.
- LSDB docs on HATS structure and lazy data access patterns (for implementation alignment):
  - https://docs.lsdb.io/en/latest/data-access/hats.html
  - https://docs.lsdb.io/en/latest/tutorials/catalog_object.html
  - https://github.com/astronomy-commons/lsdb
