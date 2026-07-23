# Spec: Streaming Training Mode (`train_stream`)

## Context

Hyrax just merged streaming **inference** as the `infer_stream` verb: a model is
loaded once and batches are processed on demand from an open-ended `IterableDataset`
(e.g. `KafkaStreamDataset`), driven through a context-manager session object
(`InferStreamSession`). This spec adds the training counterpart — a **`train_stream`**
verb that trains a model on a live/streaming data source using the same
context-manager–like form:

```python
with hy.train_stream() as session:
    for batch, metrics in session:      # metrics == {"loss": ...} (or None for a skipped batch)
        print(metrics["loss"])
        if converged:
            session.stop()
trained_model = session.close()          # weights already persisted
```

The data-plumbing layer is already streaming-aware and verb-neutral, so this is
primarily a **verb-side** addition mirroring the `infer_stream` pattern. The key
difference from batch `train`: a stream has no epochs, no length, and no
train/validate/test splits, so we cannot use Ignite's `create_trainer` /
`trainer.run(loader, max_epochs=...)` path — we drive a per-batch loop ourselves,
exactly as `InferStreamSession.__iter__` does for inference.

## Why the existing infrastructure fits

- `create_process_func("train_batch", device, model, config)`
  (`src/hyrax/pytorch_ignite.py`) returns a partial of `_inner_loop` with signature
  `(engine, batch)`. Called with `engine=None` per batch, it runs `prepare_inputs`,
  moves tensors to device, and calls the model's `train_batch`.
- The model's `train_batch` **already owns the full step** —
  `zero_grad → forward → criterion → loss.backward → optimizer.step`, returning
  `{"loss": ...}` (see `src/hyrax/models/hyrax_cnn.py`). So training a batch is just
  calling `process_func(None, batch)` **without** `torch.no_grad()` and with the model
  in `.train()` mode. This is the only substantive behavioral delta from
  `InferStreamSession.process` (which wraps the call in `torch.no_grad()`).
- Streaming data is automatic: `setup_dataset(config, splits=("train_stream",))`
  → `_build_data_provider` routes any `IterableDataset` `dataset_class` to
  `StreamingDataProvider`; `dist_data_loader` has the iterable branch
  (`batch_size=None`, `num_workers=0`, `collate_fn=dataset.collate`). No changes there.
- Model pre-flighting from a stream works via `setup_model(config, provider)` (peeks one
  sample) or `setup_model_from_sample(config, sample_batch)` — both used by `infer_stream`.

## Design decisions

- **Weight saving:** periodic checkpoint every N batches *and* a final save on close.
- **Metrics:** training loss only (no validation stream for now).
- **Code sharing:** mirror the `infer_stream` pattern; no shared-base refactor now.
- **Visibility:** full MLflow + TensorBoard logging from the start.
- **Partial batches:** train every non-empty batch as-is, with a defensive empty-batch
  skip and an optional `min_batch_size` guard (default off). See below.

## Partial / variable-sized batches

A streaming source yields **ragged batches**: `KafkaStreamDataset.__iter__` flushes
whatever it has when `batch_flush_timeout` fires during a quiet period, so batches smaller
than `batch_size` are normal. It guards with `if batch: yield batch`, so it **never yields
an empty batch**, but any non-empty size (down to 1) reaches training.

This is safe for correctness: PyTorch's batch dimension is dynamic (no shape error), and the
model's `train_batch` uses the criterion at default `reduction="mean"`, so a short batch just
averages over fewer samples. The built-in models use no BatchNorm, so size-1 batches don't
break them. Two nuances handled/documented:

- **Gradient weighting is per-batch, not per-sample.** Because the model owns
  `optimizer.step()` inside `train_batch`, a 1-sample timeout batch steps the optimizer as
  hard as a full batch. Accepted training-quality nuance for streaming, not a bug.
- **User models with BatchNorm** can error on a size-1 batch in `.train()` mode — hence the
  optional `min_batch_size` guard.

**Approach:** train every non-empty batch as-is. `TrainStreamSession.process()` computes the
sample count from the collated batch (`len(batch["object_id"])`), skips a zero-row batch
(returns `None`, no optimizer step), and — when `min_batch_size` is set — skips batches with
fewer than that many samples. No gradient accumulation (it would require bypassing the
model's self-contained `train_batch`; can be added later if needed).

## Files to create / modify

### 1. New verb: `src/hyrax/verbs/train_stream.py`
Two classes mirroring `src/hyrax/verbs/infer_stream.py`:

**`TrainStream(Verb)`** — decorated `@hyrax_verb`:
- `cli_name = "train_stream"`, `add_parser_kwargs = {}`, `description`,
  `REQUIRED_DATA_GROUPS = ("train_stream",)`, `OPTIONAL_DATA_GROUPS = ()`.
- `setup_parser` — no-op. `run_cli` — raise `NotImplementedError` (programmatic API only).
- `run(self, sample_batch=None) -> TrainStreamSession`:
  - Build the model from a configured `[data_request.train_stream]` stream
    (`setup_dataset` → `provider` → `setup_model(config, provider)` →
    `data_loader = dist_data_loader(provider, config)`), **or** from an explicit
    `sample_batch` via `setup_model_from_sample`.
  - `model.train()`; if `config["train_stream"]["model_weights_file"]`,
    `load_model_weights(config, model, "train_stream")` **before** wrapping.
  - `results_dir = create_results_dir(config, "train_stream")`;
    `init_tensorboard_logger(log_dir=results_dir)`; `log_runtime_config(...)`.
  - `wrapped_model = auto_model(model)`; `process_func =
    create_process_func("train_batch", idist_device(), wrapped_model, config)`
    (keep the unwrapped `model` for `.save()` / `.optimizer`).
  - **MLflow run spanning the session** (mirror `train.py`): tracking DB
    `sqlite:///<results_root>/mlflow/mlflow.db`, `set_experiment`, resolve `run_name`,
    `mlflow.start_run(...)`, log static params (results dir, `config["model"]`,
    batch_size, criterion + optimizer + their params). The run is **ended in `close()`**.
  - Return `TrainStreamSession(process_func, model, config, results_dir,
    close_tensorboard_logger, data_loader=..., provider=...)`.

**`TrainStreamSession`** — mirror `InferStreamSession`:
- `__iter__` — iterate `data_loader`, `yield batch, self.process(batch)`; raise
  `RuntimeError` if no data source.
- `process(batch) -> dict | None` — guard `_closed`; skip empty / sub-`min_batch_size`
  batches (return `None`); else `result = self._process_func(None, batch)` **without**
  `torch.no_grad()`; increment `_batch_count`; log each metric to TensorBoard
  (`training/training/{m}`) and MLflow (`training/{m}`) stepped by `_batch_count`; if
  `save_weights_every` set and `_batch_count % save_weights_every == 0`, `save_weights()`;
  return `result`.
- `save_weights()` — `self._model.save(results_dir / weights_filename)`.
- `stop()` — forward to `provider.stop()` if present.
- `close() -> model` — idempotent; `stop()`, final `save_weights()`, `mlflow.end_run()`
  (if a run is active), close TensorBoard logger, set `_closed`; return the trained model.
- `__enter__` / `__exit__` (calls `close()`).

### 2. Register: `src/hyrax/verbs/__init__.py`
Add `from hyrax.verbs.train_stream import TrainStream` and `"TrainStream"` to `__all__`.
No changes to `verb_registry.py`, `hyrax.py`, or `hyrax_cli/main.py` (decorator +
`Hyrax.__getattr__` auto-expose `hy.train_stream()`).

### 3. Config: `src/hyrax/hyrax_default_config.toml`
Add `[train_stream]`:
```toml
[train_stream]
model_weights_file = false      # warm-start weights, or false to train from scratch
weights_filename = "example_model.pth"
save_weights_every = false      # checkpoint every N batches, or false = only on close
experiment_name = "notebook"    # MLflow experiment
run_name = false                # MLflow run name, or false = results-dir name
min_batch_size = false          # skip batches smaller than this, or false = train all
```
`false` is the "unset" sentinel; no Pydantic validation added to this section.

### 4. Tests: `tests/hyrax/test_train_stream.py`
- Data-source-driven iteration over a `FakeConsumer`-backed `KafkaStreamDataset` +
  `HyraxLoopback`: iterate the session, assert `(batch, metrics)` yielded, weights file
  written after `close()`, model returned.
- Manual session (directly constructed with a spy `process_func`): empty batch skipped,
  sub-`min_batch_size` batch skipped, normal batch processed; `process()` after `close()`
  raises; session without a data source is not iterable; `close()` idempotent.
- `run_cli` raises `NotImplementedError`.

## Reused (unchanged)
`StreamingDataProvider`, `dist_data_loader` iterable branch, `KafkaStreamDataset`,
`setup_dataset` / `setup_model` / `setup_model_from_sample`, `create_process_func`,
`create_results_dir`, `log_runtime_config`, `load_model_weights`, TensorBoard logger,
MLflow wiring pattern from `train.py`.

## Explicitly out of scope
No validation/test stream, no `create_splits`, no `WeightedRandomSampler`; no Ignite
`create_trainer` / `max_epochs` epoch loop; no `on_epoch_start` dispatch; no shared
`StreamSession` base refactor of `infer_stream.py`; no gradient accumulation.

## Verification
1. `python -m pytest tests/hyrax/test_train_stream.py -m "not slow"`.
2. E2E smoke: configure a tiny `IterableDataset` (or `KafkaStreamDataset` with a fake
   consumer) under `[data_request.train_stream]`, iterate `hy.train_stream()`, confirm
   loss prints, weights file appears, and (with `save_weights_every` set) periodic
   checkpoints and an MLflow `training/loss` series exist.
3. `hyrax train_stream` (CLI) raises `NotImplementedError`.
4. `ruff check src/ tests/ && ruff format src/ tests/`; `pre-commit run --all-files`.
</content>
