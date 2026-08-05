# Plan: `LSDBStreamDataset` — stream HATS catalogs into `train_stream` / `infer_stream`

Status: **implemented**. This document is the design plan reconciled with what shipped;
the "Deviations found during implementation" section at the end records where the built
code differs from the original plan and why.

---

## Context

`train_stream` and `infer_stream` exist and work, but the only streaming dataset that
feeds them is `KafkaStreamDataset`. Hyrax's actual astronomy data lives in HATS catalogs
read through LSDB, and `lsdb.streams.CatalogStream` / `InfiniteStream` already yield
partitions lazily — but nothing connects the two. The only LSDB-backed dataset,
`HyraxHATSDataset`, calls `catalog.compute()` and materializes the whole thing in memory,
which defeats the purpose for large catalogs.

`LSDBStreamDataset` is a `torch.utils.data.IterableDataset` that wraps an LSDB stream and
yields fixed-size batches, so a HATS catalog — including a crossmatched or queried one
built interactively in a notebook — can be trained on or inferred over via
`with hy.train_stream() as session:` without ever holding the full catalog in memory.

Two things make this non-trivial and drive most of the design:

1. **LSDB chunks are partition-sized and ragged.** A real catalog yields 9 rows, then 10,
   then 171. Verified locally: a 12-row test catalog gives 5 partitions of sizes
   `[2, 3, 2, 3, 2]`, and a Gaia DR3 partition holds ~200k rows. Batches must be buffered
   to a fixed `batch_size`.
2. **Derived catalogs can't be a URL.** `gaia.crossmatch(tess).crossmatch(sdss)` only
   exists in memory, and it cannot be stored in the config because `log_runtime_config`
   (`config_utils.py:746`) serializes the whole runtime config with `tomlkit.dumps`.

Decisions confirmed with the user before implementation: class name `LSDBStreamDataset`;
catalog specified via HATS path/URL **or** an `lsdb://<name>` handle into an in-memory
registry; stream selected by a `stream_type` config key; nested/ragged columns out of
scope for v1; `lsdb` promoted to a runtime dependency but imported lazily.

---

## The contract to satisfy

`StreamingDataProvider` (`streaming_data_provider.py:92-95`) constructs the stream as
`dataset_cls(config=..., data_location=...)` and needs exactly four things:

| Member | Requirement |
|---|---|
| `__iter__()` | yields `list[dict]` — batches of **flat** sample dicts. The provider's `_structure()` does `np.asarray(sample[field])` and `str(sample[primary_id_field])`. |
| `peek_sample() -> dict` | one flat sample, **not lost** (replayed into the first batch). Used by `setup_model` to pre-flight the model. |
| `stop()` | ends iteration; called by session `close()` / `__exit__`. |
| `__len__` | must *exist* (`dataset_registry.py:115-119`); may raise. |

`kafka_stream_dataset.py` is the reference implementation — it establishes the
lazy-resource pattern, the `threading.Event` stop flag, the `self._buffered` replay list,
`try/finally` teardown inside `__iter__`, and `super().__init__(config, metadata_table=None)`
called **last**.

---

## 1. `src/hyrax/datasets/lsdb_stream_dataset.py`

```python
class LSDBStreamDataset(HyraxDataset, torch.utils.data.IterableDataset):
```

Module-level state:

```python
LSDB_URI_PREFIX = "lsdb://"
_CATALOG_REGISTRY: dict[str, object] = {}
```

**Imports.** `lsdb` is a declared runtime dependency, but imports stay *inside* methods to
match the convention in `hats_dataset.py:26` and `lsst_dataset.py:156`. Because the
dependency is guaranteed, a plain `import lsdb` is used — no `try/except ImportError`
ceremony. The HATS spatial-index column name comes from
`hats.pixel_math.spatial_index.SPATIAL_INDEX_COLUMN` (verified `== "_healpix_29"`) rather
than being hardcoded. It is needed in exactly one place: the `discard()` in
`_requested_columns_from_config`, which keeps `_healpix_29` out of the `columns=` list
passed to `lsdb.open_catalog` when someone sets `primary_id_field = "_healpix_29"`. It is
the DataFrame *index*, not a column, so requesting it is meaningless. `_flatten_index`
reads `frame.index.name` dynamically and never needs the constant.

### `__init__(self, config, data_location=None)`

1. `ds_config = config["data_set"]["LSDBStreamDataset"]` — **hardcode the literal**, not
   `type(self).__name__`, so user subclasses read the same defaults instead of needing
   their own config block. (Kafka does this at `kafka_stream_dataset.py:76`; it is why
   `_HookedStream` works in `test_streaming_data_provider.py:141`.)
2. Reject `data_location in (None, False)`, naming both accepted forms.
3. Keep the **raw** `data_location` string — the data-request scan compares against it.
4. `stream_type` ∈ `{"catalog", "infinite"}`, else `ValueError`.
5. `partitions_per_chunk` — `ValueError` if `is False` or `< 1` (landmine L2).
6. `seed` — `None if seed is False else int(seed)`. **`is False`, never `not seed`** (L1).
7. `batch_size` from `config["data_loader"]["batch_size"]`, validated positive.
8. `requested_columns = self._requested_columns_from_config(config)`, which also stashes
   `self._primary_id_field` and `self._explicit_fields`.
9. `self._catalog = self._resolve_catalog(ds_config, requested_columns)`.
10. State: `_stop = threading.Event()`, `_iterator = None`, `_exhausted = False`,
    `_buffered = []`, `_peek_index = 0`, `_row_checked = False`.
11. `super().__init__(config, metadata_table=None)`.

### Catalog registry

```python
@classmethod register_catalog(cls, name, catalog) -> str   # returns "lsdb://<name>"
@classmethod unregister_catalog(cls, name) -> None
@classmethod clear_catalogs(cls) -> None
@classmethod registered_catalogs(cls) -> list[str]
@classmethod _lookup_catalog(cls, name)
```

`register_catalog` validates that `name` is a non-empty string with no whitespace or `/`
(so it round-trips through the URI), checks `isinstance(catalog, lsdb.Catalog)` — otherwise
the failure surfaces much later inside `CatalogStream.__init__` — warns on overwrite, and
**returns the URI** so notebooks read:

```python
data_location = LSDBStreamDataset.register_catalog("gaia_mmu", catalog)
```

`_lookup_catalog`'s `KeyError` lists the registered names and shows the fix. The registry
is process-local and is not inherited by DataLoader workers; this is safe only because
`dist_data_loader` forces `num_workers = 0` for iterable datasets (see L8).

### `_resolve_catalog(ds_config, requested_columns)`

- `lsdb://` → **strip the prefix directly**. Do not use `urlparse().netloc`, which applies
  host/port semantics and would mangle any name containing a colon.
- Otherwise `lsdb.open_catalog(data_location, **open_catalog_kwargs)`, injecting
  `columns=requested_columns` when the user did not set `columns` — same shape as
  `hats_dataset.py:23-28`.
- **Column pruning is skipped for registered catalogs.** `crossmatch` renames columns
  (`ra_gaia_x_tess_lightcurve`), so a `fields`-derived `columns` list would be wrong; the
  user already chose the columns when building the catalog.

### `_requested_columns_from_config(config)`

Adapted from `hats_dataset.py:53-80` with three changes:

- `definition.get("data_location")`, not `[...]` — a streaming request may omit it
  entirely (`test_streaming_data_provider.py:28-34`).
- `_same_location()` replaces `Path(...).resolve()`: raw-string equality first, `False` if
  either side contains `://`, `Path` comparison only as a fallback.
- No `join_field` — `StreamingDataProvider` rejects joined datasets outright
  (`streaming_data_provider.py:59-63`).

`type(self).__name__` is still used for the *data_request* scan, which is correctly
subclass-aware since the request names the subclass.

### `_make_stream()` — the monkeypatch seam for tests

Guards `self._catalog.npartitions < 1` first (L3), then:

- `"infinite"` → `InfiniteStream(cat, partitions_per_chunk=..., seed=...)`.
  **Never pass `shuffle`** — verified: `InfiniteStream.__init__() got an unexpected keyword
  argument 'shuffle'`. A `logger.warning` fires when `shuffle = false` is set but ignored.
- `"catalog"` → `CatalogStream(cat, partitions_per_chunk=..., shuffle=..., seed=...)`.

### `_ensure_iterator()`

Creates and memoizes a single iterator. Load-bearing: `CatalogStream.__iter__` hands back
a *fresh* iterator with a *spawned* RNG on every call, so `peek_sample` and `__iter__` must
share one or peeked rows would come from a different traversal than the batches.

### `_flatten_index(frame)`

`reset_index()` only when `frame.index.name` is set and not already a column. Verified:
lsdb chunks are `NestedFrame`s with `index.name == "_healpix_29"`, dtype `int64[pyarrow]`.
This makes `primary_id_field = "_healpix_29"` usable — the only sane id for a crossmatched
catalog with no natural one. `HyraxHATSDataset` never exposes it.

### `peek_sample()`

Pulls whole chunks into `_buffered` until `_peek_index` is satisfiable, returns
`_buffered[_peek_index]`, and increments. A `while` (not `if`) tolerates a zero-row
partition, and the cursor means repeated peeks return **distinct** rows with nothing
consumed twice. Raises `RuntimeError` on `StopIteration` (empty catalog) or if stopped
first.

### `_check_first_row(row)` — runs once

- If `_primary_id_field` is known and missing, raise a `RuntimeError` naming the available
  columns. Otherwise the user gets a bare `KeyError` four frames deep in
  `streaming_data_provider.py:129`.
- If `fields` was not set, `logger.warning` that every column becomes a model input —
  including `_healpix_29` and the `coord_ra`/`coord_dec` that lsdb re-adds even when pruned.

### `__iter__()` — the buffering algorithm

```python
iterator = self._ensure_iterator()

batch = list(self._buffered)          # replay peeked rows ahead of the first batch
self._buffered = []
self._peek_index = 0

frame = None                          # current chunk, tracked for the finally block
taken = 0

try:
    while len(batch) >= self.batch_size:      # establish len(batch) < batch_size (L4)
        full, batch = batch[: self.batch_size], batch[self.batch_size :]
        yield full

    while not self._stop.is_set():
        try:
            chunk = next(iterator)
        except StopIteration:                 # PEP 479 — MANDATORY, see L5
            self._exhausted = True
            frame, taken = None, 0
            break

        frame = self._flatten_index(chunk)
        taken, n_rows = 0, len(frame)
        while taken < n_rows:
            need = self.batch_size - len(batch)
            window = frame.iloc[taken : taken + need]
            rows = window.to_dict(orient="records")
            if rows and not self._row_checked:
                self._check_first_row(rows[0])
            batch.extend(rows)
            taken += len(window)
            if len(batch) >= self.batch_size:
                full, batch = batch, []       # clear BEFORE yielding
                yield full
        # No early exit on self._stop here: a chunk already fetched is always drained.

    if batch:                                 # only the final batch may be short
        final, batch = batch, []
        yield final
finally:
    leftover = frame.iloc[taken:].to_dict(orient="records") if frame is not None else []
    if batch or leftover:
        self._buffered = batch + leftover + self._buffered
    if self._exhausted or self._stop.is_set():
        self._iterator = None
```

| Event | Behavior |
|---|---|
| peeked rows | delivered at the head of the first batch, in peek order |
| chunk > `batch_size` | split via `frame.iloc[taken:taken+need]`; remainder carries forward |
| chunk < `batch_size` | accumulates; nothing emitted until `batch_size` rows exist |
| finite stream exhausted | `break` → single short final batch flushed |
| `stop()` mid-chunk | current chunk **fully drained**, then exit; no new chunk fetched |
| consumer `break`s out | `GeneratorExit` at a `yield` → `finally` re-buffers the partial batch *and* the undrained chunk tail; iterator survives, next pass resumes in place |
| re-iterate after exhaustion | fresh iterator, new spawned RNG → a second full epoch |

Three details that must not be "simplified" away:

- The `full, batch = batch, []` swap **before** each `yield` (Kafka's idiom at
  `kafka_stream_dataset.py:220`). Without it, a `GeneratorExit` at the last `yield` would
  re-buffer an already-consumed batch and duplicate rows on the next pass.
- The final flush lives in the `try` body, **never** in `finally` — yielding while
  unwinding a `GeneratorExit` raises `RuntimeError: generator ignored GeneratorExit`.
- The `finally` must push back `frame.iloc[taken:]` as well as `batch`. See deviation D2.

### `stop()` — and its honest limitation

`self._stop.set()`. `stop()` **cannot interrupt a blocked `next(iterator)`**: with
`client=None` LSDB computes the chunk inline (`dask.compute()`), with a distributed client
it blocks in `Future.result()`, and lsdb exposes no `close()`, no cancellation, and no
timeout knob. So `stop()` takes effect at the next chunk boundary, worst case one chunk
compute. Kafka has the same shape but bounds it with `batch_flush_timeout`; lsdb has no
equivalent.

In practice this never bites: `TrainStreamSession.close()` (`train_stream.py:364`) calls
`stop()` on the same thread *after* iteration has ended. For `stream_type = "infinite"`,
the documented way to end a session is to `break` out of the loop — that is immediate. A
smaller `partitions_per_chunk` bounds the window.

### `__len__()`

`raise TypeError("LSDBStreamDataset is a stream and has no length.")` — defined only to
satisfy `__init_subclass__`; the iterable branch of `dist_data_loader` never calls it.

### DataFrame → `list[dict]`: `to_dict(orient="records")`

- `itertuples` is disqualified — it mangles non-identifier names (`mag-r` → `_3`), and
  `mag-r` is exactly what `test_hats_dataset.py:19` uses; crossmatched catalogs are full of
  such names.
- `.iloc[i].to_dict()` per row builds a `Series` per row — much slower.
- Verified: pyarrow-backed columns unwrap to native `int`/`float`/`str`/`list`, so
  `np.asarray` in the provider gets clean values with no pyarrow scalars leaking.
- Converting per **batch remainder** rather than per chunk keeps the Python-object working
  set to one batch instead of one partition.

---

## 2. Config — `src/hyrax/hyrax_default_config.toml`

Inserted after `[data_set.KafkaStreamDataset]`, before `[data_loader]`:

```toml
[data_set.LSDBStreamDataset]
# Which lsdb stream to build over the catalog:
#   "catalog"  -> lsdb.streams.CatalogStream, a single finite pass over every partition.
#                 Best for inference: every object is visited exactly once and the run
#                 ends on its own when the catalog is exhausted.
#   "infinite" -> lsdb.streams.InfiniteStream, endless random resampling of partitions.
#                 Best for training: batches keep arriving until you stop the session,
#                 so training is not bounded by a single pass over the catalog.
stream_type = "catalog"

# Number of HATS partitions computed per chunk. Larger values amortize per-chunk dask
# overhead but hold more rows in memory; lsdb clips this to the catalog's partition count.
partitions_per_chunk = 1

# Shuffle the partition order and the rows within each chunk. Ignored (and always on)
# when stream_type = "infinite"; lsdb's InfiniteStream does not accept this option.
shuffle = true

# Random seed for partition/row shuffling. Set to false for fresh entropy each run.
seed = false

# Extra keyword arguments passed directly to lsdb.open_catalog. Ignored when
# data_location is an "lsdb://<name>" reference to a pre-registered catalog.
[data_set.LSDBStreamDataset.open_catalog_kwargs]
```

No config migration (those are only for renames) and no pydantic schema (`data_set` is not
in `PYDANTIC_VALIDATED_KEYS`). `lsdb://gaia_mmu` survives
`DataRequestConfig.resolve_data_location` (`data_request.py:58-66`), which passes through
anything matching `<scheme>://`.

---

## 3. Registration — `src/hyrax/datasets/__init__.py`

Import order is deliberate (`# ruff: noqa: I001`, drives autoapi ordering). Added after the
`KafkaStreamDataset` import and to `__all__`:

```python
from .lsdb_stream_dataset import LSDBStreamDataset
```

Registration itself is automatic via `HyraxDataset.__init_subclass__`.

---

## 4. Dependency

`lsdb` moved from the `dev` extra into `[project] dependencies` alongside
`confluent-kafka`. Three dataset classes now need it, and this also makes `hats` reachable
(it previously arrived only transitively). The stale comment
`# Used to test lsst dataset classes` was removed.

The tradeoff is explicit: every hyrax install now pulls hats/dask/distributed/
nested-pandas/healpy/astropy. That is a real increase over the previous default and a
resolver-conflict surface worth watching in CI.

---

## 5. Tests — `tests/hyrax/test_lsdb_stream_dataset.py`

45 tests. Plain `import lsdb`, **not** `pytest.importorskip` — now that lsdb is a runtime
dep, a missing install should fail loudly rather than silently skip the suite.

Fixtures: an autouse `clear_catalogs()` teardown; a `tiny_catalog` of 12 rows built with
`lsdb.from_dataframe` (verified `npartitions == 5`, chunk sizes `[2, 3, 2, 3, 2]`, so
`batch_size = 5` must give `[5, 5, 2]`); a `hats_catalog_path` written via `write_catalog`;
and a `_with_chunks` helper that monkeypatches `_make_stream` to return a list of plain
DataFrames, exercising the buffering logic without touching dask.

Coverage:

- **Construction/config** — missing `data_location`; `__len__` raises; invalid
  `stream_type`; `partitions_per_chunk` ∈ `{False, 0, -1}`; `seed = 0` survives while
  `seed = false` becomes `None`; default config block has all five keys.
- **Registry** — URI round-trip; resolution to the same object; unknown name lists known
  names; non-`Catalog` and unusable names rejected; unregister/clear; `lsdb://` survives
  `DataRequestConfig.model_validate`.
- **Buffering** — exact `batch_size` across ragged chunks (3/7/2 @ 5 → `[5,5,2]`); one
  11-row chunk @ 4 → `[4,4,3]`; exact multiple yields no short batch; every batch but the
  last is full; empty chunk tolerated; `peek_sample` loses no row and appears first; two
  peeks return distinct rows and both replay; peek on empty stream raises; `stop()` flushes
  the partial; `stop()` before iteration yields nothing; **break-then-resume loses no rows
  and duplicates none**; second pass after exhaustion restarts; missing `primary_id_field`
  raises actionably; omitted `fields` warns.
- **Real lsdb streams** — finite pass gives `[5,5,2]` and 12 unique ids; `InfiniteStream`
  does not exhaust under `islice`; `stream_type="infinite"` + `shuffle=false` constructs
  without `TypeError` and logs the warning; `_healpix_29` promoted and usable as
  `primary_id_field`; `mag-r` survives as a dict key.
- **HATS path** — column pruning from `fields`; no `fields` reads everything;
  `open_catalog_kwargs` passed through; registered catalogs skip pruning.
- **End to end** — through `StreamingDataProvider` + `dist_data_loader`
  (`loader.batch_size is None`, batch shapes); `hy.train_stream()` sees every id exactly
  once and writes weights and `runtime_config.toml`; `hy.infer_stream()` likewise; and the
  declarative HATS-path variant.

---

## 6. Docs

- **`docs/pre_executed/lsdb_stream_dataset.ipynb`** — pre-executed against live Gaia DR3
  with outputs saved, linked from `docs/common_workflows.rst`. `docs/pre_executed/` exists
  precisely so nbsphinx publishes notebooks without their deps installed on ReadTheDocs.
  Shows: open + query, `register_catalog` (and *why* the registry exists), a small inline
  dense autoencoder over catalog columns, `train_stream` on an infinite stream, a 202,682-row
  raw LSDB chunk becoming uniform 1024-row batches, a finite pass over a 30-row catalog
  giving `8, 8, 8, 6` with every object seen exactly once, and the known limitations.
- **`docs/dataset_class_reference.rst`** — new "Streaming datasets" section. The page
  previously documented only the map-style `__len__` + `get_<field>` contract, which
  neither `KafkaStreamDataset` nor this class fits.
- **`docs/conf.py`** — added `(r"^py:.*", r"^lsdb\..*")` to `nitpick_ignore_regex`,
  matching the existing "packages that have their own docs" entries, so the `lsdb.Catalog`
  parameter type does not fail the warnings-as-errors sphinx build.

---

## 7. Landmines (all verified locally)

- **L1 — `seed = 0` is falsy.** The TOML sentinel is `false` but `0` is a legitimate seed.
  Use `seed is False`. Same trap for `partitions_per_chunk`.
- **L2 / L3 — `partitions_per_chunk = false` silently degenerates.** `CatalogStream` does
  `min(partitions_per_chunk, npartitions)`; verified `min(False, 5) == 0`, and
  `partitions_left[:-0] == []` while `partitions_left[-0:]` is *everything* — the whole
  catalog computes as one chunk. Validate `>= 1`. Separately, a 0-partition catalog dies as
  `ValueError: No objects to concatenate` deep inside dask; guard `npartitions < 1`.
- **L4 — over-full replay buffer can infinite-loop.** If `_buffered` holds ≥ `batch_size`
  rows on entry, `need <= 0` makes the `iloc` window empty and `taken` never advances. The
  pre-flush `while` at the top of `__iter__` prevents this.
- **L5 — PEP 479.** An uncaught `StopIteration` inside a generator becomes
  `RuntimeError: generator raised StopIteration`, turning every clean end-of-catalog into a
  crash. The `try/except StopIteration: break` is mandatory.
- **L6 — no `>>>` in any docstring.** `addopts = "--doctest-modules --doctest-glob=*.rst"`
  with `testpaths` including `src`: pytest imports every module under `src/` and *executes*
  every doctest it finds. A `>>> lsdb.open_catalog(...)` example would run for real during
  collection and hit the network. Use `.. code-block:: python`.
- **L7 — ragged nested columns fail as a numpy `ValueError`, not Hyrax's `RuntimeError`.**
  The `RuntimeError` branch at `data_provider.py:314-324` only fires when shapes are
  *equal*; ragged shapes fall through to `np.array(values)` →
  `setting an array element with a sequence... inhomogeneous shape`. The docstring names
  that exact symptom and points at `collate_<field>`. Flip side: **fixed-length** nested
  columns already work and stack to `(N, k, n_cols)`.
- **L8 — `num_workers = 0` is load-bearing, not a Kafka quirk.** A plain `IterableDataset`
  does no worker sharding, so N workers would each build their own stream and emit every
  row N times.
- **L9 — no real prefetch with `client=None`.** LSDB's `submit_next_partitions` calls
  `.compute()` inline, so partition I/O never overlaps GPU work. A `use_dask_client` config
  key is a reasonable follow-up.
- **L10 — don't mutate `config` in `__init__`.** `DataProvider._apply_configurations`
  returns `base_config` *itself* when there is no `dataset_config`.

---

## 8. Deviations found during implementation

**D1 — `stop()` must not exit mid-chunk.** The first implementation broke out of the inner
loop as soon as `_stop` was set after a yield, which silently discarded the remaining rows
of a chunk that had *already been fetched and computed*. For inference that means objects
skipped with no indication. The inner early-exit was removed: a fetched chunk is always
drained to completion, and `stop()` takes effect on the outer loop by not fetching a new
chunk. Caught by `test_stop_ends_iteration_and_flushes_partial`.

**D2 — the `finally` block must push back the undrained chunk tail, not just `batch`.** The
original plan only re-buffered `batch`. When a consumer `break`s out of the session loop,
`GeneratorExit` arrives at a `yield` where `batch` has just been cleared — so the rows
between `taken` and the end of the current chunk were dropped entirely. With chunks of 6
and `batch_size = 4`, a break after the first batch lost rows 4 and 5 permanently. Fixed by
tracking `frame`/`taken` outside the `try` and pushing back
`batch + frame.iloc[taken:].to_dict(orient="records")`. Caught by
`test_break_then_resume_loses_no_rows`.

**D3 — `lsdb.ConeSearch` catalogs cannot be streamed (upstream limitation).** In lsdb
0.10.0, `open_catalog(..., search_filter=lsdb.ConeSearch(...))` produces a catalog whose
own stream classes raise
`ValueError: Selected Pixel Order: 2, Pixel: 0 not found in operation`. This affects
`CatalogStream` directly, not anything Hyrax does, and it holds with or without a
subsequent `.query()`. `query()` on an unfiltered catalog streams fine. Documented in the
notebook with `query()` as the workaround; worth an upstream issue.

**D4 — `batch_size` validation added.** Not in the original plan, but the same `false`
sentinel hazard as `partitions_per_chunk` applies, and a zero would make the inner window
empty forever.

---

## Verification

1. `python -m pytest tests/hyrax/test_lsdb_stream_dataset.py -v` — 45 passed.
2. `python -m pytest -m "not slow" tests src` — 579 passed. Also confirms the
   `--doctest-modules` collection of the new module is clean.
3. `ruff check` and `ruff format --check` on the new files — clean.
4. `pre-commit run --files <changed>` — all hooks pass except the sphinx build, which fails
   only on 22 warnings that predate this work (`splitting_utils`, `kafka_stream_dataset`,
   `pytorch_ignite`, `hyrax_parquet_dataset`). This change adds none.
5. Manual end-to-end against live Gaia DR3 via the notebook: all batches exactly
   `batch_size`, the finite run ends cleanly on exhaustion with no traceback,
   `runtime_config.toml` is written (proving nothing unserializable reached the config), and
   weights land in the results dir.
