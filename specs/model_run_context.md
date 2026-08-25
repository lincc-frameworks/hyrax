# Run context for Hyrax models

## TL;DR

Formalize the informal `context` dict that already existed for vector databases and the
ONNX exporter into a single process-wide run context (`src/hyrax/context.py`), and make it
reachable from models via `get_context()`. Models can now write arbitrary artifacts into
the run's results directory. No model, dataset, or verb signature changes — existing models
are untouched and opt in by calling `get_context()`.

## Motivation

Hyrax models could only reach run-scoped state through `self.config`. They had no idea
where the run's results directory was, so a user wanting to persist something that doesn't
fit the TensorBoard/MLflow scalar-metric paradigm — attention maps, per-epoch embeddings,
custom diagnostics, a JSON of notes — had nowhere to put it.

A `context` dict already existed for exactly this purpose, but only for vector databases
and the ONNX exporter. It was unstructured, had inconsistent value types (`str` in one call
site, `Path` in another), and the copy in `infer.py` was dead code — orphaned when
vector-db insertion moved to `save_to_database.py`.

## Decisions

- **Delivery is a module-global accessor**, not a constructor argument. Mirrors the existing
  `get_tensorboard_logger()` / `get_trace()` patterns. Keeps `@hyrax_model`, `setup_model`,
  and every model signature unchanged.
- **Contents are `results_dir`, `verb`, `rank`, `world_size`** (plus `ml_framework` for ONNX
  export). `rank` matters because under distributed training every rank shares one results
  directory.
- **Models write to `results_dir` directly.** No dedicated subdirectory, no path helper.
- **Scope includes cleaning up the pre-existing context**, so there is one concept rather
  than three.

## The key mechanic: mutate one dict, never rebind it

`get_context()` must be safe to call *before* the context is populated, because
`infer_stream` built its model before its results dir existed, and because a model's
`__init__` may want to stash the handle.

The tensorboard logger solves this with a stateless proxy class plus `__getattr__`. For a
dict the same late-binding comes far more cheaply: keep **one module-level dict object for
the life of the process** and only ever `clear()` / `update()` it in place. A handle taken
at any time stays valid, and `get_context()` returns a real `dict`, so it drops straight
into the existing `VectorDB.__init__(config, context)` and `export_to_onnx(...)` call sites.

This is load-bearing. Assigning a new dict to the module global would silently break the
late-binding guarantee.

## `src/hyrax/context.py`

```python
class ContextKeys(TypedDict, total=False):
    """Documents the keys Hyrax installs. A type-annotation aid, not enforced."""
    results_dir: Path   # this run's results directory
    verb: str           # "train" | "infer" | "test" | "infer_stream" | "onnx" | ...
    rank: int           # idist rank; 0 when not distributed
    world_size: int     # idist world size; 1 when not distributed
    ml_framework: str   # ONNX export only


class RunContext(dict):
    """A plain dict that explains itself when a key is missing."""
    def __missing__(self, key): ...


# One object for the life of the process. Never rebound - only cleared and
# updated in place - so handles taken before init_context() stay valid.
_context = RunContext()


def get_context() -> RunContext: ...                              # returns _context
def init_context(results_dir, verb, **extra) -> RunContext: ...   # clear + populate
def update_context(**kwargs) -> RunContext: ...                   # merge, preserve
def clear_context() -> None: ...
```

- `init_context` coerces `results_dir` to `Path`; `update_context` fills `rank` /
  `world_size` from a **function-local** `import ignite.distributed as idist`, keeping the
  module cheap for user model code to import.
- `RunContext` subclasses `dict` rather than being a `Mapping` or proxy, so
  `isinstance(ctx, dict)`, `**ctx`, and pickling all keep working.
- `get_context` is re-exported from `hyrax/__init__.py`.

## Verb wiring

`create_results_dir` is the single marker for "a run just started". Each verb calls
`init_context(results_dir, "<verb>")` immediately after it:

| Verb | Note |
|---|---|
| `train.py` | plus the distributed repair below |
| `infer.py` | **deleted** the dead `context = {}` / `context["results_dir"] = ...` |
| `test.py` | |
| `infer_stream.py` | `create_results_dir` **moved above** `setup_model` |
| `to_onnx.py` | `init_context(output_dir, "onnx", ml_framework="pytorch")`, replacing the hand-built dict |
| `save_to_database.py` | replaces `{"results_dir": str(vector_db_path)}` |
| `create_splits.py`, `engine.py`, `reduce_dimensions.py` | uniformity; no models involved |

**`infer_stream` reordering.** It built the model before creating its results dir, unlike
every other model-building verb (train and infer both create the dir before even setting up
the dataset). Moving `create_results_dir` above the model setup makes it consistent and
makes `results_dir` available inside `__init__`. Cost: a failed `setup_dataset` now leaves
an empty results dir behind — already the behavior of every other verb.
`log_runtime_config` stays where it was, since it intentionally runs after
`load_model_weights` so the resolved weights path is captured.

**`database_connection.py` keeps building its context explicitly.** It connects to a
*pre-existing* database directory rather than starting a run; calling `init_context` there
would clobber the context of whatever run just finished.

### Distributed repair

`idist.Parallel` spawns fresh processes, which re-import modules — so the context is empty
in every child rank. `Train._training` already receives `rank` and `results_dir`, so it
repopulates there, on the path every rank passes through:

```python
update_context(
    results_dir=results_dir, verb="train",
    rank=idist.get_rank(), world_size=idist.get_world_size(),
)
```

`update_context` rather than `init_context` so the single-process path (where `_training`
is called directly) doesn't wipe keys a model added during `setup_model`.

## Model-side contract

`model_registry.py`, `pytorch_ignite.py`, and every model in `src/hyrax/models/` are
**unchanged**. Models opt in:

```python
from hyrax import get_context

@hyrax_model
class MyModel(nn.Module):
    def __init__(self, config, data_sample=None):
        super().__init__()
        self.config = config
        self.context = get_context()   # late-binding handle, safe to take here
        self.activations = []

    def train_batch(self, batch):
        ...
        self.activations.append(hidden.detach().cpu().numpy())
        return {"loss": loss.item()}

    def train_post_epoch(self):
        rank = self.context["rank"]
        np.save(self.context["results_dir"] / f"activations_rank{rank}.npy",
                np.concatenate(self.activations))
        self.activations.clear()
```

The natural flush points are the existing duck-typed zero-arg hooks `train_post_epoch` and
`validate_post_epoch` in `pytorch_ignite.py`, which were previously undocumented.

Users may also add their own keys. The context is cleared at the start of each verb run, so
user keys last for that run only.

## Cleanups included

1. **Deleted the dead context** in `infer.py`.
2. **Standardized `results_dir` on `Path`.** `save_to_database.py` was the only site passing
   a `str`. Safe: every ChromaDB use coerces via `str(results_dir)`, and
   `database_connection.py` already handed Qdrant a `Path`. Added a matching `str()`
   coercion at `qdrantdb_impl.py:27` so both backends are explicit. `model_exporters.py`
   uses the `/` operator and *requires* a `Path`.
3. **Renamed `ctx` → `context`** in `export_to_onnx`; it is called positionally, so no
   caller broke.
4. **Typed the explicit consumers** with `ContextKeys` (`VectorDB.__init__`,
   `vector_db_factory`, `export_to_onnx`) and replaced the vague *"an instance of the
   context object"* docstring with the real key list.

Not in scope: the `#!` TODO at `qdrantdb_impl.py:58-61` asking for a vector-size context
key. `ContextKeys` makes it a clean follow-up.

## Files

**New:** `src/hyrax/context.py`, `tests/hyrax/test_context.py`, `specs/model_run_context.md`

**Modified:** `src/hyrax/__init__.py`, `src/hyrax/model_exporters.py`,
`src/hyrax/vector_dbs/{vector_db_interface,vector_db_factory,qdrantdb_impl}.py`, verbs
(`train`, `infer`, `test`, `infer_stream`, `to_onnx`, `save_to_database`,
`database_connection`, `create_splits`, `engine`, `reduce_dimensions`),
`docs/model_class_reference.rst`, `HYRAX_GUIDE.md`

## Verification

`tests/hyrax/test_context.py` covers:

- `get_context()` returns the *same object* before and after `init_context` — the
  late-binding guarantee `infer_stream` depends on.
- `init_context` clears prior keys; `update_context` preserves them.
- `results_dir` coerced to `Path` from both `str` and `Path`.
- Missing keys raise a `KeyError` naming the available keys and explaining that the context
  is empty outside a verb run.
- A `@hyrax_model` model built outside any verb gets an empty context — proving direct
  construction (as many existing tests do) did not regress.
- **End-to-end**: a model with a `train_post_epoch` that writes JSON to
  `self.context["results_dir"]`; `h.train()`; assert the file lands in the train results dir.
- Each verb leaves the context pointing at its own results directory.

Full fast suite: 625 passed. Two pre-existing collection errors are unrelated
(`test_range_read_lsst_dataset.py` needs `fitsio`; stale `docs/**/results/` dirs from
previously-run notebooks are picked up by collection).
