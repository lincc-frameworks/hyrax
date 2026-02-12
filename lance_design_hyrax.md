# Lance Storage Layer — Summary for Hyrax Developers

## What and Why

Inference results (`infer`, `umap`, `test`, `engine`) currently live as batched `.npy`
files managed by `InferenceDataSet` / `InferenceDataSetWriter`. We're replacing the
storage backend with [Lance](https://lancedb.github.io/lance/), a columnar format with
faster random-access reads than Parquet or raw NumPy. See
[#428](https://github.com/lincc-frameworks/hyrax/issues/428).

This is **not** a format swap bolted onto `InferenceDataSet`. It's a clean break:

- **`ResultDatasetWriter`** — new Lance-only writer.
- **`ResultDataset`** — new Lance-only reader, integrated with HyraxQL (`get_data`,
  `get_object_id` getters auto-discovered by `DataProvider`).
- **`InferenceDataSet`** — stays read-only for `.npy`; deprecated over two releases.

Both live in a new `result_dataset.py` alongside the existing `inference_dataset.py`.

## Key Design Decisions

1. **No dual-backend class.** No `if lance: ... else: npy: ...` branching. The two
   storage formats are two separate classes.

2. **Incremental writes.** `write_batch()` calls `table.add()` per batch inside the
   training loop; `commit()` runs `table.optimize.compact_files()` at the end. No
   in-memory accumulation.

3. **`original_config` / `original_dataset` bridging is removed.** `ResultDataset` does
   not carry metadata about the original survey dataset. Users who need both combine them
   via `DataProvider` / `data_request` config.

4. **Standard `(config, data_location)` constructor.** `ResultDataset` plugs into
   `data_request` the same way any other `HyraxDataset` does — no special `verb` parameter.

5. **Factory-based selection (Lance-only for now).** A writer factory is wired into
   `infer`, `test`, `umap`, and `engine` and always produces a Lance-backed
   `ResultDatasetWriter` (there is currently no `storage_format` knob or `.npy` writer
   fallback). A reader factory is responsible for opening Lance results; legacy `.npy`
   outputs continue to be handled by `InferenceDataSet` via auto-detection on disk
   (`lance_db/` subdirectory present → Lance, otherwise `.npy`).

6. **LanceDB files isolated in `lance_db/` subdirectory** inside the results dir, avoiding
   collisions with existing artifacts like `original_dataset_config.toml`.

## Rollout Plan

| Step | Deliverable |
|------|-------------|
| 0 | Add `lancedb` + `pyarrow` dependencies |
| 1 | `ResultDatasetWriter` — batch-incremental Lance writes |
| 2 | `ResultDataset` — Lance reader with HyraxQL getters |
| 3 | Wire writer into `infer`, `test`, `umap`, `engine` via factory |
| 4 | Wire reader into `umap`, `save_to_database`, `lookup` via factory |
| 5 | `hyrax convert_results` CLI verb for `.npy` → Lance migration |
| 6 | Deprecate `InferenceDataSet` (soft warning now, removal next release) |

Each step is independently testable. No big-bang cutover.

## What Breaks / What's Deferred

- **`visualize` verb will not work with Lance results.** It depends on the
  `original_config` / `metadata()` bridging that `ResultDataset` intentionally drops.
  Refactoring `visualize` to use `DataProvider` for metadata is a separate effort. Until
  then it stays `.npy`-only and emits a clear error if pointed at Lance results.

- **`batch_num` column is dropped.** It was a `.npy` layout artifact; Lance has no
  concept of it.

## Risks Worth Knowing

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Performance regressions in unrelated benchmarks** (ChromaDB/Qdrant slowdowns seen in prototype) | Medium | May be import-time overhead from `pyarrow`/`lancedb`. Run benchmarks in isolation. |
| **`lancedb` API churn** (young project, API has shifted between versions) | Medium | Pin minimum version; add CI job against latest release. |
| **PyArrow version conflicts** with `pandas`, `astropy`, `chromadb` | Medium | Check compatibility matrix before pinning. |
| **`__getitem__` edge cases** (prototype hang in `test_nan.py` from missing `IndexError`) | High | Thorough edge-case tests: out-of-range, negative indices, empty results, iteration termination. |
| **Floating-point fidelity in `.npy` → Lance migration** (NaN, Inf, denormals, float16) | Low | Migration script includes element-wise verification pass. |
| **Disk size overhead** (Lance columnar metadata on large embedding vectors) | Low | Benchmark realistic tensor sizes; Lance compression available if needed. |
