# Plan: Streaming Inference Mode (`infer_stream`)

## Context

Hyrax's `hyrax.infer()` is a blocking call — it sets up a DataLoader and runs until the configured dataset is exhausted. Users want to run inference **continuously**: load the model once, accept batches of data arriving at irregular intervals (long gaps are fine), process each batch as it arrives, accumulate results, and stop only when explicitly closed. This fits the "live telescope feed" or "interactive notebook" use case where data arrives in bursts over a long session.

---

## Approach

Add a new `InferStream` verb whose `run()` method performs one-time setup (model load + weights + results writer) and returns an `InferStreamSession` context manager. The user calls `session.process(batch)` for each batch and `session.close()` (or uses `with`) when done.

**User-facing API:**
```python
hy = Hyrax()
with hy.infer_stream(sample_batch=representative_batch) as session:
    for batch in my_data_source:          # may have long pauses
        results = session.process(batch)  # returns CPU tensor, saves to Lance
# session.close() auto-called; returns ResultDataset
```

Or without context manager:
```python
session = hy.infer_stream(sample_batch=first_batch)
session.process(batch1)
session.process(batch2)
result_dataset = session.close()
```

**Batch format** (same as existing infer pipeline):
```python
{
    "object_id": ["id1", "id2", ...],      # required by Lance writer
    "data": {"image": np.ndarray, ...}     # model-specific fields
}
```

---

## Files to Create / Modify

### 1. New: `src/hyrax/verbs/infer_stream.py`

A `@hyrax_verb`-decorated `InferStream(Verb)` class with `cli_name = "infer_stream"` and `REQUIRED_DATA_GROUPS = (infer_stream)`.

`run(self, sample_batch: dict | None = None) -> InferStreamSession`:
1. `create_results_dir(config, "infer_stream")` — timestamped output dir
2. `init_tensorboard_logger(log_dir=results_dir)`
3. `setup_model_from_sample(config, sample_batch)` — new helper (see #3 below)
4. `model.eval()` + `idist.auto_model(model)` + `torch.set_default_device(device.type)`
5. `load_model_weights(config, model, "infer_stream")`
6. `log_runtime_config(config, results_dir)` + `model.save(results_dir / "inference_weights.pth")`
7. `create_save_batch_callback(results_dir)` — reuses existing closure
8. Build `process_func = _create_process_func("infer_batch", device, wrapped_model, config)` — reuses existing partial builder
9. `return InferStreamSession(process_func, save_batch_callback, config, results_dir)`

CLI stub: raises `NotImplementedError("infer_stream is a programmatic API; use hyrax.infer_stream() in Python/notebook.")`

Also define `InferStreamSession` in this same file (no separate file needed):

```python
class InferStreamSession:
    def __init__(self, process_func, save_batch_callback, config, results_dir): ...
    def process(self, batch: dict) -> torch.Tensor:
        # guard: raise RuntimeError if closed
        # with torch.no_grad(): result = self._process_func(batch)
        # self._save_batch(batch, result)
        # return result.detach().cpu()
    def close(self) -> ResultDataset:
        # self._save_batch_callback.data_writer.commit()
        # self._closed = True
        # close_tensorboard_logger()
        # return load_results_dataset(config, results_dir)
    def __enter__(self): return self
    def __exit__(self, *_): self.close(); return False
```

### 2. Modify: `src/hyrax/pytorch_ignite.py`

Add a new public helper after `setup_model` (around line 155):

```python
def setup_model_from_sample(config: dict, sample_batch: dict) -> torch.nn.Module:
    """Like setup_model but accepts a pre-formatted batch dict instead of a DataProvider.
    Used by InferStream to pre-flight model architecture without a dataset.
    """
    from hyrax.trace import reset_trace
    model_cls = fetch_model_class(config)
    prepared_sample = model_cls.prepare_inputs(sample_batch)
    retval = model_cls(config=config, data_sample=prepared_sample)
    reset_trace()
    return retval
```

Also export `_create_process_func` by renaming to `create_process_func` (remove leading underscore) so it can be imported cleanly by `infer_stream.py`. Update its one internal callsite in `create_engine` accordingly.

### 3. Modify: `src/hyrax/verbs/__init__.py`

Add `InferStream` import and export:
```python
from hyrax.verbs.infer_stream import InferStream  # after existing imports
```
Add `"InferStream"` to `__all__`.

### 4. Modify: `src/hyrax/hyrax_default_config.toml`

Add after the `[infer]` section:
```toml
[infer_stream]
# Path to model weights. If false, uses the most recent training results.
model_weights_file = false
```

---

## Functions to Reuse (with locations)

| Function | File | Lines | How used |
|---|---|---|---|
| `create_results_dir` | `config_utils.py` | ~622 | Called once in `run()` |
| `init_tensorboard_logger` / `close_tensorboard_logger` | `tensorboardx_logger.py` | — | Open in `run()`, close in `close()` |
| `setup_model_from_sample` | `pytorch_ignite.py` | new | Replaces `setup_model` for streaming |
| `load_model_weights` | `models/model_utils.py` | ~10 | Called once in `run()` |
| `log_runtime_config` | `config_utils.py` | — | Called once in `run()` |
| `create_save_batch_callback` | `pytorch_ignite.py` | 958 | Called once; reused across all `.process()` calls |
| `create_process_func` (renamed) | `pytorch_ignite.py` | 540 | Builds per-batch partial in `run()` |
| `_inner_loop` | `pytorch_ignite.py` | 516 | Called indirectly via `create_process_func` partial |
| `extract_model_method` | `pytorch_ignite.py` | 570 | Called inside `create_process_func` |
| `idist.device()` / `idist.auto_model()` | ignite | — | Device setup in `run()` |
| `load_results_dataset` | `datasets/result_factories.py` | ~36 | Called in `close()` |
| `ResultDatasetWriter.commit()` | `datasets/result_dataset.py` | 118 | Called in `close()` |

---

## Key Design Decisions

- **No new Dataset class needed** — data arrives as pre-formatted dicts, bypassing DataLoader/DataProvider entirely.
- **`torch.no_grad()`** — must be explicit in `process()` since `create_evaluator`'s engine machinery is bypassed.
- **`torch.set_default_device()`** — must be called in `run()` (as `create_engine` normally does at line 566) before the `process_func` partial is built.
- **Thread safety** — `process()` is not thread-safe; document this. No lock needed for MVP.
- **`sample_batch` is required** — if `None`, raise `ValueError` with a clear message. The config-based bootstrap option is deferred.

---

## Verification

1. **Unit tests** in `tests/hyrax/test_infer_stream.py`:
   - `test_session_returns_from_run` — verify `hyrax.infer_stream(sample_batch=x)` returns `InferStreamSession`
   - `test_process_returns_tensor` — call `.process(batch)`, verify output tensor shape
   - `test_multiple_batches_accumulate` — call `.process()` N times, verify `result_dataset` has all IDs after `.close()`
   - `test_context_manager` — verify `close()` is called on exit
   - `test_use_after_close_raises` — verify `RuntimeError` on `.process()` after `.close()`
   - `test_matches_regular_infer` — feed same batches to both `infer()` and `infer_stream()`; assert Lance outputs match

2. **Notebook smoke test** — open `infer_stream`, pass a few batches, call `close()`, inspect `result_dataset`.

3. **Linter/tests:** `ruff check src/ tests/` + `python -m pytest -m "not slow"` pass cleanly.
