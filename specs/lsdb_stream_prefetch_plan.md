# Unblock LSDB streaming: overlap chunk fetch with training

## Status

Phases 1–5 are implemented and committed to the working tree on branch
`claude/train-stream-verb-planning-8qjyzm`. What remains is measurement and one
end-to-end correctness check, both of which need a real catalog and better hardware
than the laptop this was written on. See "Remaining work" at the end.

| Phase | State |
|---|---|
| 1. Instrumentation | Done — DEBUG timing in the dataset and the provider, plus `scripts/bench_lsdb_stream.py` |
| 2. Dask client wiring | Done |
| 3. Background prefetch thread | Done |
| 4. Tests | Done — 85 pass across the two streaming suites, 743 across `tests/` |
| 5. Docs & config | Done, except the pre-executed notebook (deliberately untouched) |

Files changed:

- `src/hyrax/datasets/lsdb_stream_dataset.py` — the substance
- `src/hyrax/datasets/streaming_data_provider.py` — structuring cost timing
- `src/hyrax/hyrax_default_config.toml` — three new keys
- `tests/hyrax/test_lsdb_stream_dataset.py` — parametrization + new tests
- `docs/dataset_class_reference.rst` — why streams prefetch and why Kafka must not
- `scripts/bench_lsdb_stream.py` — new diagnostic script

---

## Context

`train_stream` on an LSDB-backed catalog is data-starved. The whole pipeline is strictly
serial and runs on one thread:

```
next(lsdb_iterator)  →  _flatten_index + to_dict  →  _structure per row  →  collate  →  GPU step
```

`dist_data_loader` forces `num_workers = 0` for iterable datasets
(`src/hyrax/pytorch_ignite.py`, the `isinstance(dataset, IterableDataset)` branch), so
PyTorch's own `prefetch_factor` machinery is unavailable — there was **zero** prefetch depth.
The GPU idled for the entire duration of every chunk compute.

**The key finding:** lsdb already implements exactly the prefetch that was asked for, and
Hyrax was disabling it. In `lsdb/streams/catalog_streams.py`, `CatalogIterator.__next__`
submits the *next* chunk's future before returning the current one — but
`submit_next_partitions` only returns a real async `Future` when a `client` is passed. With
`client=None` it returns `_FakeFuture(delayed.compute())`, which computes **synchronously
inline**. `LSDBStreamDataset._make_stream()` never passed `client=`, so every `next()` blocked
for a full chunk compute — even in `docs/pre_executed/basic_lsdb_stream_test.ipynb`, which
does create a `Client(n_workers=2)`.

Note also: calling `next(iterator)` early *inline* (the originally sketched approach) does not
help — it blocks the training loop for the same duration, just sooner. The win has to come
from concurrency, not reordering.

**Intended outcome:** chunk fetch and per-row decoding happen off the training thread, so
training-step time — not I/O — sets the batch rate.

Explicitly **not** in scope: a cross-chunk shuffle buffer; `num_workers > 0` for streams;
applying any of this to `KafkaStreamDataset` (its `__iter__` commits offsets only on generator
resumption, so prefetching ahead of the consumer would silently break at-least-once delivery).

---

## Phase 1 — Measure first

`src/hyrax/trace.py` records call order, not wall-clock throughput, so it was not the right
tool.

**1a. `scripts/bench_lsdb_stream.py`** reproduces the TESS setup from cell 6 of
`docs/pre_executed/basic_lsdb_stream_test.ipynb` and drives `TrainStreamSession` manually so
the two halves are separable:

```python
it = iter(session.data_loader)
t0 = perf_counter(); batch = next(it); t1 = perf_counter()
session.process(batch);                 t2 = perf_counter()
# t1-t0 = data wait   |   t2-t1 = train step
```

Two details the first attempt got wrong, both since fixed:

- **Warmup is required.** `setup_model` peeks a sample, which pulls a whole chunk before the
  timer starts. Without `--warmup`, the first batches come free out of that buffer and the run
  looks as though it never fetched anything (`chunk fetch 0.00 s over 0 chunks`).
- **The seed must be pinned.** With `stream_type = "infinite"` each config draws different
  random partitions, and partition-size variance is larger than the effect being measured.
  `--seed` (default 17) pins the draw so configs are comparable.

**1b. In-dataset breakdown** — `_StreamTimings` splits the cost into producer-side *fetch* and
*convert*, and consumer-side *wait*. Emitted at DEBUG every `TIMING_LOG_EVERY_CHUNKS`:

```
lsdb stream after 20 chunks (10077 rows): fetch 68.06 s | convert 6.40 s |
consumer wait 77.71 s | 135 rows/s produced
```

The wait/fetch gap is the diagnostic: when prefetching works, consumer wait collapses toward
zero even though fetch does not. `StreamingDataProvider.__iter__` logs its structuring cost the
same way, every `STRUCTURE_LOG_EVERY_BATCHES`.

### What was measured

On a Mac (MPS, TESS over HTTPS), 40 batches x 64, `partitions_per_chunk = 1`:

| | data wait | chunk fetch | rows decoded |
|---|---|---|---|
| prefetch=0 | 77.7 s (67%) | 68.1 s / 2 chunks | 10,077 |
| prefetch=2 | 110.3 s (69%) | 141.1 s / 2 chunks | 23,891 |

**This is not evidence that prefetch is slower.** One chunk is 5–12k rows and 35–70 s, so 40
batches barely crosses one boundary, and the two runs drew different random partitions — this
was the run that motivated adding `--seed`. Two signals are real and worth keeping:

- Data wait is ~67% of wall clock. The stream is the bottleneck, as suspected.
- Chunk fetch dwarfs row decode (68 s vs 6.4 s), so Phase 2 targets the right stage.

---

## Phase 2 — Wire the dask Client through to lsdb

New config keys under `[data_set.LSDBStreamDataset]`:

```toml
use_dask_client = true
dask_client_address = false
```

- `_resolve_dask_client()` returns `None` when `use_dask_client` is false; otherwise
  `Client(dask_client_address)` when an address is set, else `dask.distributed.get_client()`
  inside `try/except ValueError` (raised when no default client exists) → `None` with an
  actionable INFO log.
- Called lazily from `_make_stream()`, not `__init__` — the notebook creates its `Client`
  around the `train_stream()` call, and `_make_stream` is not reached until first iteration.
- `client=` is passed to both `InfiniteStream` and `CatalogStream`.
- `_warn_on_small_chunks()` warns when `partitions_per_chunk < 2 * len(workers)`; lsdb
  recommends at least 2x, and the default is 1.
- No new dependency: `dask.distributed` is already a hard dependency of `lsdb`.

**Process-vs-thread tradeoff**, documented in the module docstring. A default `LocalCluster`
uses processes, so every chunk is serialized back to the client — a real cost for the nested
frames light-curve catalogs produce. `Client(processes=False)` gives async futures with
in-process results and no serialization, and is usually the right choice for single-machine
notebook training.

---

## Phase 3 — Background prefetch thread (fetch + row conversion)

Gives depth-N buffering (lsdb's built-in prefetch is depth-1) and moves the pandas→dict
transpose off the training thread. Torch releases the GIL during forward/backward, so a
Python-heavy producer thread overlaps usefully.

```toml
prefetch_chunks = 2
```

`_ChunkPrefetcher` owns the lsdb iterator, a `queue.Queue(maxsize=prefetch_chunks)`, and one
daemon thread. The producer runs `next(iterator)` → `_flatten_index(chunk).to_dict("records")`
→ `put(rows)`.

Design points that matter:

- **Errors are queued, not logged.** A producer exception is put on the queue and re-raised on
  the consumer's thread; otherwise the traceback dies with the thread and the failure looks
  like an empty stream.
- **A sentinel is emitted on every exit path** (via `finally`), so a consumer blocked in
  `next_rows()` always learns the producer is gone.
- **`_put` waits in short slices** rather than blocking outright, so an abandoned producer
  cannot wedge itself against a full queue nobody is draining.
- **Two flags, not one.** `_shutdown` asks the producer to stop after the chunk it holds
  (nothing lost); `_abandoned` additionally lets it discard that chunk.
- **`shutdown(wait=...)` keeps draining while joining**, because the producer may be blocked in
  `_put` and only a consumer taking items lets it reach its exit. `_release_source()` passes
  `wait=False` for `infinite` streams — they resample forever, so an in-flight chunk carries
  nothing a later one cannot, and waiting would stall every `break` out of a training loop for
  as long as a fetch takes. A finite `catalog` stream always waits.

`_SyncChunkSource` is the `prefetch_chunks = 0` path, behind the same
`next_rows`/`drain`/`shutdown` interface so `__iter__` has one code path.

Other changes to `LSDBStreamDataset`: `_ensure_source()` (lazily, so a dataset stopped before
iteration starts no thread); `peek_sample()` pulls from the source too, preserving the
one-iterator-owner constraint; `_check_first_row` stays on the consumer side so validation
errors surface where the caller can see them; `close()` releases a thread started by a peek
that was never iterated.

**The invariant that makes prefetch safe on by default:** the consumer loop still checks
`self._stop.is_set()` *before* pulling a chunk, and the `finally` pushes back
`rows[taken:] + self._release_source()`. So which rows are yielded, in what order, is
unchanged — the producer only changes *when* chunks compute. Nothing fetched is ever dropped,
which is what keeps `stream_type = "catalog"` inference exact.

All 45 pre-existing tests in `test_lsdb_stream_dataset.py` passed unchanged with
`prefetch_chunks = 2`, including `test_stop_ends_iteration_and_flushes_partial` (asserts
`[4, 2]`) and `test_break_then_resume_loses_no_rows`. If a future change breaks those, the
design is wrong, not the test.

---

## Phase 4 — Tests

In `tests/hyrax/test_lsdb_stream_dataset.py`. The existing `_with_chunks` helper monkeypatches
`_make_stream` to a plain list of DataFrames, which works unchanged with a prefetcher.

- A `prefetch_chunks` fixture parametrized over `(0, 2)` with ids `sync`/`prefetch`, threaded
  through 12 buffering tests. Proving output is identical either way is the whole safety
  argument.
- `test_prefetch_runs_ahead_of_the_consumer` — asserts the producer reaches chunk 3 while the
  consumer has taken one batch, rather than inferring it from a timing measurement.
- `test_prefetched_but_unconsumed_rows_are_pushed_back` — break mid-stream at depth 3, resume,
  assert all 24 rows exactly once.
- `test_producer_exception_reaches_the_consumer`.
- `test_prefetch_thread_does_not_leak`, parametrized over both stream types, covering
  exhaustion, `break`, `stop()`, and `close()`.
- `test_close_releases_a_thread_started_by_peek`.
- Three client-wiring tests: forwarded when resolvable, `None` when absent, and `get_client()`
  never called when `use_dask_client = false`.
- `test_default_config_block` updated; `prefetch_chunks` validation tests for negative values
  and for the `false`/`0` equivalence.

No config migration needed — these are new keys, not renames.

---

## Phase 5 — Docs & config

- `[data_set.LSDBStreamDataset]` in `hyrax_default_config.toml` — three keys, commented in the
  existing style, including the `Client(processes=False)` tip.
- Module docstring of `lsdb_stream_dataset.py` — a "Keeping the consumer fed" section covering
  `use_dask_client`, `prefetch_chunks`, and `partitions_per_chunk`, and how they stack.
- `docs/dataset_class_reference.rst` — why `num_workers = 0` means streams must hide latency
  themselves, and why `KafkaStreamDataset` deliberately does not prefetch.

---

## Remaining work

1. **Controlled benchmark.** Same seed, both configs, enough batches to reach steady state
   (~300 at batch 64, roughly 6 min each on the laptop):

   ```
   python scripts/bench_lsdb_stream.py --batches 300 --warmup 10 --client threads --prefetch 0
   python scripts/bench_lsdb_stream.py --batches 300 --warmup 10 --client threads --prefetch 2
   ```

   Then sweep `{none, threads, processes} x prefetch {0, 2, 4}` — `--sweep` does the client
   axis. Success is consumer wait dropping toward zero, or the train step becoming the
   binding constraint.

2. **Correctness under prefetch, on a real catalog.** Run `infer_stream` with
   `stream_type = "catalog"`, `shuffle = false`, prefetch on and off; assert both runs produce
   the identical `object_id` sequence. This is the check that the push-back actually holds
   against real lsdb chunks rather than test fixtures.

3. **The notebook.** `docs/pre_executed/basic_lsdb_stream_test.ipynb` was deliberately left
   alone — it is pre-executed with committed outputs, so editing the source without re-running
   would desync them. Once the benchmark says which client mode wins, re-run cell 7 with that
   setting and compare tqdm it/s against the recorded baseline.

## Known unrelated breakage on this branch

- `tests/hyrax/test_train_stream.py::test_train_stream_iterates_streaming_dataset` fails. It
  fails on the pristine tree too — a Kafka mock dataset with no `get_*` methods, so
  `StreamingDataProvider._structure` raises `KeyError: 'image'`. Pre-existing, not addressed.
- `pre-commit` fails on the Sphinx build: 9 warnings, treated as errors, all pre-existing
  (`kafka_stream_dataset`, `hyrax_ts2vec`, `models`, `verbs`, `reduce_dimensions`). Verified by
  stashing the changes and re-counting.
- `python -m pytest -m "not slow"` from the repo root hits 16 collection errors from stray
  `prepare_inputs.py` files under `docs/*/results/`. Scope to `tests/` to avoid them.
