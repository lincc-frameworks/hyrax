# Lance Storage Layer Design Document

## Context

Hyrax currently stores inference results (from `infer`, `umap`, `test`, `engine` verbs) as
NumPy structured arrays on disk (`batch_0.npy`, `batch_1.npy`, ..., `batch_index.npy`). These
are written by `InferenceDataSetWriter` and read by `InferenceDataSet`.

Issue [#428](https://github.com/lincc-frameworks/hyrax/issues/428) proposes adopting the
[Lance](https://lancedb.github.io/lance/) columnar format, which offers better random-access
read performance than Parquet and a familiar columnar data model. A prototype exists on branch
`dtj-lance-opt` (PR #666) that modifies `InferenceDataSet` in-place with dual-backend support.

The [issue comment](https://github.com/lincc-frameworks/hyrax/issues/428#issuecomment-3856034643)
expands the scope beyond a format swap to include:

1. Creating a new `ResultDataset` class (not just renaming `InferenceDataSet`)
2. Integrating `ResultDataset` with HyraxQL (getter-based data access via `DataProvider`)
3. Removing the `original_config` / `original_dataset` metadata-bridging machinery
4. Providing an `.npy`-to-Lance migration script
5. Deprecating `InferenceDataSet` over time

This document describes a stepwise implementation plan.

---

## Guiding Principles

- **Incremental delivery.** Each step produces a working, testable state. No big-bang rewrites.
- **No dual-backend in one class.** The prototype's approach of `if lance: ... else: npy: ...`
  throughout `InferenceDataSet` leads to high complexity. Instead, `ResultDataset` is a new,
  Lance-only class, and `InferenceDataSet` stays as-is for `.npy` reads.
- **Sunset .npy files.** When the Hyrax user community has migrated forward and fully
  adopted the LanceDB format, and used the CLI migration tool to convert their existing
  `.npy` file sets, we can stop reading `.npy` files.
- **Write Lance incrementally.** Per the issue discussion, use `table.add()` inside the batch
  loop and `table.optimize()` at the end, rather than accumulating all data in memory.
- **Use the LanceDB async API if needed.** The LanceDB docs warn against wrapping calls in
  threads; the library already uses threads and async I/O internally. Where parallelism is
  needed, use the recommended `lancedb` async API rather than `multiprocessing.Pool`.
- **Minimal config plumbing.** Writer and reader classes do not receive `config` as a
  constructor parameter. The factory at the call site reads `config` to select the class;
  the writer gets the original dataset config from `original_dataset.config`.

---

## Step 0: Preparatory Refactors

**Goal:** Add dependencies and prepare the ground for new classes.

Add `lancedb` and `pyarrow` to `pyproject.toml` dependencies, as the prototype does.

---

## Step 1: `ResultDatasetWriter` — Lance-based Writer

**Goal:** A new writer class that writes results to Lance format, one batch at a time.

### Design

Create `src/hyrax/data_sets/result_dataset.py` containing `ResultDatasetWriter`.

```python
class ResultDatasetWriter:
    def __init__(self, result_dir)
    def write_batch(self, object_ids, data)
    def commit(self)   # finalizes: calls table.optimize(), writes config
```

Key details:
- Constructor takes only `result_dir`.
- On first `write_batch`, create a LanceDB connection to `results_dir/lance_db/` via
  `lancedb.connect(result_dir / "lance_db")` and create the table via `db.create_table()`
  with an explicit PyArrow schema derived from the first tensor's dtype and shape.
- On subsequent `write_batch` calls, use `table.add()` to append data incrementally.
  This avoids accumulating all data in memory (a flaw in the prototype).
- `write_index()` calls `table.optimize()` to consolidate fragments,
  then writes `original_dataset_config.toml` (reusing existing logic from
  `InferenceDataSetWriter`).
- No multiprocessing pool. LanceDB uses its own internal async I/O. If benchmarks show
  this is a bottleneck, the LanceDB async API (`AsyncConnection`, `AsyncTable`) can be
  adopted later.

### Schema

The Lance table `results` will have these columns:

| Column      | Type                              | Notes                                   |
|-------------|-----------------------------------|-----------------------------------------|
| `object_id` | `string`                          | Object ID, stored as string             |
| `data`      | `fixed_size_list(<dtype>, N)`     | Flattened tensor-like data of length N  |

The `batch_num` column from the prototype is **not included** — it is an artifact of the
`.npy` batch-file layout and has no meaning in Lance.

Using `fixed_size_list` rather than variable-length list enables Lance to use a more compact
and faster layout for uniform-length vectors.

### Tensor Storage

Multi-dimensional tensors are flattened to 1D for storage. Shape and dtype metadata are
stored in the Arrow schema's metadata dict:

```json
{"tensor_shape": "[2, 3]", "tensor_dtype": "float32"}
```

On read, tensors are reshaped and cast accordingly. The dtype is derived from the first
tensor in the first batch. All tensors in a given table must share the same dtype and shape.

This approach supports arbitrary tensor dtypes (float16, float32, float64) without hardcoding
assumptions, which is important for a general-purpose framework.

---

## Step 2: `ResultDataset` — Lance-based Reader

**Goal:** A new reader class that loads Lance results and exposes them via the HyraxQL
getter interface.

### Design

Create `ResultDataset` in `src/hyrax/data_sets/result_dataset.py`.

```python
class ResultDataset(HyraxDataset, Dataset):
    def __init__(self, config, data_location)
    def __len__(self)
    def __getitem__(self, idx)      # returns torch.Tensor (for backward compat)
    def get_data(self, idx)         # getter for HyraxQL
    def get_object_id(self, idx)    # getter for HyraxQL
```

Key details:
- Constructor signature is `(config, data_location)` — the standard two-arg pattern used
  by all HyraxDataset subclasses. `data_location` points to the results directory. This
  allows `ResultDataset` to be used directly in `data_request` config:
  ```toml
  [data_request.infer.results]
  dataset_class = "ResultDataset"
  data_location = "./results/infer_20250201_120000"
  ```
- Opens a `lancedb.connect(data_location / "lance_db")` and `db.open_table("results")`.
- `__len__` returns `table.count_rows()`.
- `__getitem__` uses `table.take_offsets([idx])` for O(1) random access. Supports single int,
  list, slice, and numpy array indexing. Returns `torch.Tensor`.
- Must raise `IndexError` for out-of-range indices (critical — the prototype had a bug here
  that caused `test_nan.py` to hang).
- `get_data(idx)` and `get_object_id(idx)` are the HyraxQL getter methods. `DataProvider`
  will auto-discover these via its `get_*` introspection. These are the only getters;
  additional getters (aliases, specialized names) can be added later if needed.
- `ids()` yields IDs by scanning only the `object_id` column (projection pushdown).
- **No `original_config` / `original_dataset` machinery.** Metadata bridging to the original
  dataset is not needed — users control combined datasets via HyraxQL / `DataProvider`.
- **No `metadata()` override.** If users need metadata alongside results, they configure
  both datasets in `data_request` and `DataProvider` joins them.
- **No `verb` parameter.** `ResultDataset` does not know which verb produced it. The
  verb-based auto-discovery of the most recent results directory is the verb's responsibility,
  not the dataset's.

---

## Step 3: Wire Writer Into Verbs

**Goal:** Verbs that produce results (`infer`, `test`, `umap`, `engine`) use
`ResultDatasetWriter` for new writes.

### Approach

A factory function creates the writer:

```python
def create_results_writer(result_dir):
    return ResultDatasetWriter(result_dir)
```

Since Lance is the default format going forward, new writes always use `ResultDatasetWriter`.
No config option is needed to control the storage format.

This function is called by:
- `create_save_batch_callback` in `pytorch_ignite.py` (for `infer` and `test`)
- `Umap.run` in `verbs/umap.py`
- `Engine.run` in `verbs/engine.py`

A separate factory selects the reader by detecting what's on disk:

```python
def load_results_dataset(config, results_dir):
    if (Path(results_dir) / "lance_db" / "results.lance").exists():
        return ResultDataset(config, results_dir)
    else:
        return InferenceDataSet(config, results_dir)
```

This auto-detection on read means users can freely mix old `.npy` results and new Lance
results without changing config. Users with existing `.npy` files will continue to read
them via this auto-detection.

---

## Step 4: Wire Reader Into Verbs

**Goal:** Verbs that consume results (`umap`, `save_to_database`, `lookup`) transparently
load either `ResultDataset` or `InferenceDataSet` based on what's on disk.

### Approach

Replace direct `InferenceDataSet(...)` construction in each verb with the
`load_results_dataset(...)` factory from Step 3. Affected verbs:

| Verb              | File                          | Usage                                  |
|-------------------|-------------------------------|----------------------------------------|
| `umap`            | `verbs/umap.py`               | Reads inference results as input       |
| `save_to_database`| `verbs/save_to_database.py`   | Reads inference results for DB insert  |
| `lookup`          | `verbs/lookup.py`             | Reads inference results for ID lookup  |
| `test`            | `verbs/test.py`               | Returns results for inspection         |
| `infer`           | `verbs/infer.py`              | Returns results for inspection         |
| `engine`          | `verbs/engine.py`             | Returns results for inspection         |

### Visualize Verb — Deferred

The `visualize` verb relies on `InferenceDataSet.metadata()` and the
`original_config` / `original_dataset` bridging. Refactoring `visualize` to use `DataProvider` for
metadata is a separate effort. Until that work is done:

- `visualize` continues to use `InferenceDataSet` and works only with `.npy` results.
- If Lance results are detected, `visualize` emits a clear error message explaining that
  visualization of Lance results requires the `visualize` verb to be updated.
- This keeps the scope of the Lance work contained and avoids a risky refactor of the
  visualization pipeline.

---

## Step 5: `.npy`-to-Lance Migration Script

**Goal:** A CLI verb that converts existing `.npy` result directories to Lance format.

### Design

Add a new verb `convert_results`:

```bash
hyrax convert_results --results-dir ./results/infer_20250201_120000
```

Implementation:
1. Load `batch_index.npy` to get the ID list and batch mapping.
2. Iterate over `batch_*.npy` files, load each, and use `ResultDatasetWriter.write_batch()`
   to write records incrementally.
3. Call `write_index()` to finalize.
4. Verify the conversion: read back via `ResultDataset`, compare row count to the original
   `len(batch_index)`, and spot-check a sample of tensors for bitwise equality.
5. Print a success message. Do **not** delete the `.npy` files — let the user do that
   manually after verifying.

Given the small user base (~5 people), this can be simple and single-threaded.

---

## Step 6: Deprecation of `InferenceDataSet`

**Goal:** A gradual deprecation path.

### Phase 1 (this release): Soft deprecation
- `InferenceDataSet` continues to work for `.npy` reads.
- New writes default to Lance via `ResultDatasetWriter`.
- A `DeprecationWarning` is emitted when `InferenceDataSet.__init__` is called, advising
  users to run `hyrax convert_results` and switch to `ResultDataset`.
- Documentation is updated to describe `ResultDataset` as the primary class.

### Phase 2 (next release): Hard deprecation
- `InferenceDataSet` emits a louder warning (or is removed, depending on user migration).
- No `storage_format` configuration option is introduced; result format selection is always
  based on auto-detection of the on-disk files when reading.

---

## Step 7: Tests

### Unit Tests for `ResultDatasetWriter`

- Write batches, call `write_index()`, verify Lance table row count.
- Verify tensor round-trip fidelity (write then read, compare values).
- Verify that writing to an already-existing directory raises or overwrites appropriately.
- Verify shape and dtype metadata is stored correctly for 1D and multi-dimensional tensors.
- Verify different tensor dtypes (float32, float64, float16).

### Unit Tests for `ResultDataset`

- Read from a Lance directory written by `ResultDatasetWriter`.
- `__getitem__` with int, list, slice, numpy array.
- `__getitem__` with out-of-range index raises `IndexError`.
- `__len__` matches expected count.
- `ids()` returns all IDs.
- `get_data(idx)` and `get_object_id(idx)` return correct values.
- Chaining: write inference results, read as `ResultDataset`, write umap, read as
  `ResultDataset` — verify end-to-end.
- Usable as a `data_request` dataset class via `DataProvider`.

### Integration Tests

- `test_nan.py` — Must pass with Lance backend (the prototype had a hang here).
- `test_test.py` — Update assertions to accept Lance files.
- `test_inference_dataset.py` — The existing `test_order` test should be adapted (or a
  parallel version created) for `ResultDataset`, verifying ID and data ordering consistency.
- End-to-end: `infer` -> `umap` pipeline with Lance storage.

### Benchmark Tests

- Write performance: `InferenceDataSetWriter` vs `ResultDatasetWriter` across dataset sizes.
- Random-access read performance: `InferenceDataSet[i]` vs `ResultDataset[i]`.
- Sequential scan performance: iterating all elements.
- Memory usage: peak RSS during write and read.

---

## Resolved Ambiguities

These questions were raised during design and resolved through discussion.

| # | Question | Resolution |
|---|----------|------------|
| A1 | Table name | `results` (materializes as `lance_db/results.lance/`) |
| A2 | Tensor dtype | Store dtype in Arrow schema metadata; support arbitrary dtypes |
| A3 | `visualize` verb migration | Deferred to a separate effort; `visualize` works only with `.npy` until updated |
| A4 | `batch_num` column | Dropped; it is an artifact of the `.npy` layout |
| A5 | Getter methods | `get_data` and `get_object_id` only; more can be added later |
| A6 | Storage format selection | No config option; new writes always use Lance; auto-detection on read |
| A7 | Config threading to writer | Not needed; writer constructors stay clean |
| A8 | `ResultDataset` in `data_request` | Yes; constructor signature is `(config, data_location)` like other datasets; no `verb` parameter |

---

## Risks

### R1. Performance regressions

The PR #666 benchmark run showed regressions in unrelated benchmarks (ChromaDB insert was
2.4x slower, Qdrant 1.3x slower). These may be artifacts of the benchmark environment or
side effects of adding `lancedb`/`pyarrow` as dependencies (e.g., import-time overhead,
memory pressure). **Mitigation:** Run benchmarks in isolation on a clean environment.
Investigate whether `lancedb` import pulls in heavy dependencies that interfere with
other vector DB libraries.

### R2. `lancedb` stability and API churn

LanceDB is a relatively young project. Its API has changed between versions (e.g., the
async API, `take_offsets` method). Pinning the version is important, but may conflict
with other dependencies. **Mitigation:** Pin to a minimum version with known-good API
surface. Add a CI job that tests against the latest `lancedb` release to catch
breakage early.

### R3. Large tensor storage efficiency

Storing tensors as `fixed_size_list(<dtype>, N)` in Lance may not be as compact as raw
`.npy` for very large N (e.g., 2048-dim embeddings). Lance adds columnar overhead
(metadata, indices, null bitmaps). **Mitigation:** Benchmark disk usage for realistic
tensor sizes. If Lance files are substantially larger, consider Lance's built-in
compression options.

### R4. `test_nan.py` hang (prototype bug)

The prototype had a bug where `__getitem__` didn't raise `IndexError` for out-of-range
indices, causing infinite iteration in `test_nan.py`. This was fixed in commit `34a5bcb`
but highlights a class of bugs where Lance's API behaves differently from NumPy's.
**Mitigation:** Thorough edge-case testing for `__getitem__`, including empty results,
out-of-range indices, and negative indices. Validate that `for x in dataset:` terminates
correctly.

### R5. `visualize` verb breakage

`ResultDataset` drops the `original_config` / `original_dataset` / `metadata()` bridge. Until
`visualize` is refactored, it will not work with Lance results.
**Mitigation:** `visualize` detects Lance results and emits a clear error message
directing users to either use `.npy` results or wait for the `visualize` update.

### R6. Migration script data integrity

Converting `.npy` to Lance involves re-encoding all tensors. Floating-point round-trip
fidelity must be verified, especially for edge cases (NaN, Inf, denormals, float16).
**Mitigation:** The migration script includes a verification pass that reads back
the Lance data and compares it element-wise against the original `.npy` data.

### R7. Disk layout conflicts

Hyrax results directories already contain non-Lance artifacts (e.g.,
`original_dataset_config.toml`, model weight files from `test` verb). To avoid confusion
or conflicts, the LanceDB database is always stored in a dedicated subdirectory:
`lancedb.connect(results_dir / "lance_db")`. This keeps all LanceDB-related files
isolated from other result artifacts and ensures clean coexistence.

### R8. Thread safety during writes

The current `.npy` writer uses a `multiprocessing.Pool` with fork context. The
`ResultDatasetWriter` writes synchronously via `table.add()` in the batch loop. If this
is too slow, the LanceDB async API should be used rather than `multiprocessing`, per the
library's documentation. **Mitigation:** Benchmark synchronous writes first. Only add
async complexity if there's a measured bottleneck.

### R9. PyArrow version conflicts

Adding `pyarrow` as an explicit dependency may conflict with versions required by other
Hyrax dependencies (e.g., `pandas`, `astropy`, `chromadb`). **Mitigation:** Check
compatibility matrix before pinning. LanceDB bundles its own Lance native library but
relies on PyArrow for data interchange — ensure the PyArrow version range is compatible
across all dependencies.
