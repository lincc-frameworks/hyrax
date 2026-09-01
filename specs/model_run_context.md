# Run context for Hyrax models

## TL;DR

Formalize the informal `context` dict that already existed for vector databases and the
ONNX exporter into a single run context (`src/hyrax/context.py`), reachable from models via
`get_context()`. The `Verb` base class owns the context's lifecycle: it installs a fresh
context around every verb's `run()` and releases it when the run ends, even if `run()`
raises. Models can now write arbitrary artifacts into the run's results directory. No model,
dataset, or verb signature changes — existing models are untouched and opt in by calling
`get_context()`.

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
- **Initial contents are `results_dir`, `verb`** (plus `ml_framework` for ONNX export).
  Nothing about distributed rank is in the context — see "Model-side contract" below.
- **Models write to `results_dir` directly.** No dedicated subdirectory, no path helper.
- **The `Verb` base class owns the context's lifecycle, not each verb.**
  `Verb.__init_subclass__` wraps every verb's `run()` in a context manager, so a verb cannot
  forget to establish a context and cannot leave one behind when it finishes or raises.
  Verbs still record their own results directory themselves, once they have created it.
- **Each verb run gets a brand-new context object**, not a single long-lived one that is
  cleared and reused. This is what lets an object that outlives its run — a `VectorDB`, an
  `InferStreamSession` — simply hold on to the context from its own construction rather than
  copying values out of it or worrying about a shared global changing under it later.
- **Scope includes cleaning up the pre-existing context**, so there is one concept rather
  than three.

## The key mechanic: a fresh context object per run

Earlier drafts of this design kept one module-level dict for the life of the process and
only ever mutated it in place, on the theory that a handle taken early had to keep working
forever. That turned out to be the wrong invariant. What code that outlives its run — a
`VectorDB` handed back for interactive querying, an `InferStreamSession` — actually needs is
for the context object it holds to keep meaning *its own run*, not to have a single
long-lived object it must not be surprised by. Rebinding the module global on every run
delivers that directly: each `with run_context(...):` block installs a distinct `RunContext`
instance, so holding on to `get_context()` from inside one run's `__init__` is automatically
a snapshot of that run, and nothing later can change the values inside it.

Within a single run, `get_context()` is still safe to call before every key is populated —
it just returns whatever the current object has in it so far, same as before. What changed
is the guidance for models: **call `get_context()` at the point of use, not once in
`__init__`.** This isn't about the context being empty yet; it's about distributed training
pickling the model into per-rank child processes. A context handle taken in the parent
becomes a stale, disconnected copy once unpickled in a child, so code holding it never sees
the update the child process makes for its own rank. Reading `get_context()` fresh inside a
hook always returns the live context for whichever process is running it.

`use_context()` is the other half of the "outlives its run" story: it lets code that
captured a context earlier re-install it temporarily. `InferStreamSession.process()` uses it
so that model code driven after `infer_stream`'s own `run()` has already returned (and
released its context) still sees the run it belongs to.

## `src/hyrax/context.py`

```python
class ContextKeys(TypedDict, total=False):
    """Documents the keys Hyrax installs. A type-annotation aid, not enforced."""
    results_dir: Path   # this run's results directory
    verb: str            # "train" | "infer" | "test" | "infer_stream" | "onnx" | ...
    ml_framework: str    # ONNX export only


class RunContext(dict):
    """A plain dict that explains itself when a key is missing."""
    def __missing__(self, key): ...


# The context for the run currently underway. Rebound (not mutated) by run_context()
# and clear_context(), so a handle captured during one run keeps that run's values
# even after a later run replaces what get_context() returns.
_current: RunContext = RunContext()


def get_context() -> RunContext: ...                     # returns _current

@contextmanager
def run_context(verb, **extra): ...    # install a fresh context, release on exit

@contextmanager
def use_context(context: RunContext): ...  # re-install a captured context

def update_context(**kwargs) -> RunContext: ...  # merge into the current context
def clear_context() -> None: ...                 # reset to a fresh, empty context
```

- There is no `init_context`. `run_context` (installed automatically by the `Verb` base
  class) replaced it.
- `update_context` coerces `results_dir` to `Path` when given.
- `RunContext` subclasses `dict` rather than being a `Mapping` or proxy, so
  `isinstance(ctx, dict)`, `**ctx`, and pickling all keep working.
- `get_context` is re-exported from `hyrax/__init__.py`.

## Verb wiring

`Verb.__init_subclass__` (in `verb_registry.py`) wraps every subclass's own `run()` — not
`run_cli`, since every verb's `run_cli` delegates to `run` and wrapping both would establish
two contexts per CLI invocation — in `run_context(cli_name)`. That installs an empty
`RunContext(verb=cli_name)` on entry and releases it (`clear_context()`) in a `finally`,
including when `run()` raises.

`create_results_dir` (in `config_utils.py`) itself calls
`update_context(results_dir=directory, verb=postfix)` the moment it creates the directory.
Because of that, most verbs need no context-specific code at all — they get `results_dir`
and `verb` for free just by calling `create_results_dir`.

| Verb | Note |
|---|---|
| `train.py` | wired by `create_results_dir`; see the distributed repair below |
| `infer.py` | **deleted** the dead `context = {}` / `context["results_dir"] = ...`; `create_results_dir` covers it now |
| `test.py` | wired by `create_results_dir` |
| `infer_stream.py` | `create_results_dir` **moved above** `setup_model` |
| `to_onnx.py` | `create_results_dir(config, "onnx")` wires `results_dir`/`verb`; one extra `update_context(ml_framework="pytorch")` call |
| `save_to_database.py` | `update_context(results_dir=vector_db_path, verb="vector-db")`, replacing `{"results_dir": str(vector_db_path)}` |
| `database_connection.py` | `update_context(results_dir=vector_db_path)` — leaves `verb` as whatever `run_context` installed (`"database_connection"`) |
| `create_splits.py`, `engine.py`, `reduce_dimensions.py` | need no context-specific code at all — `create_results_dir` already covers them |

**`infer_stream` reordering.** It built the model before creating its results dir, unlike
every other model-building verb (train and infer both create the dir before even setting up
the dataset). Moving `create_results_dir` above the model setup makes it consistent and
makes `results_dir` available inside `__init__`. Cost: a failed `setup_dataset` now leaves
an empty results dir behind — already the behavior of every other verb.
`log_runtime_config` stays where it was, since it intentionally runs after
`load_model_weights` so the resolved weights path is captured.

**`database_connection.py` also calls `update_context`**, pointing it at the pre-existing
database directory it was asked to connect to. That directory is not a new run's output, but
the context is how the database finds its location, so it has to be set before the database
is built. See "Nothing takes a context parameter" below for why this is safe.

Verb runs are not expected to nest: `run_context` releases to empty rather than to whatever
was current beforehand, so a verb that calls another verb (only the deprecated `umap` verb,
which delegates to `reduce_dimensions`) is left without a context once the inner call
returns — harmless here because `umap.run()` returns immediately after delegating.

### Nothing takes a context parameter

Consumers call `get_context()` themselves rather than receiving a context argument:
`VectorDB.__init__(config)`, `vector_db_factory(config)`, `export_to_onnx(model, sample,
config)`. Whoever starts the work points the context at the right directory first.

There is one hazard this creates, and it is worth understanding before adding another
consumer. An object that outlives its run needs to hold on to the context it was given
rather than reading `get_context()` again later — because a later verb's run installs its
own new context object, and reading `get_context()` after that run has started would return
the wrong run's context entirely. `database_connection.run()` hands the database object back
to the user for interactive querying, and `chromadb_impl` re-reads the database's directory
at *query* time to spawn its `ProcessPoolExecutor` workers. If `VectorDB` read `get_context()`
fresh on every query instead of holding what it was given, this would break:

```python
db = h.database_connection()   # context -> database dir
h.train()                      # a new run installs its own context -> train results dir
db.get_by_id(...)              # would look for the database inside the train dir
```

So `VectorDB.__init__` stores `self.context = get_context()` at construction, and exposes
`results_dir` as a property (`self.context["results_dir"]`) that subclasses read. Because
each verb run gets its own distinct context object (see "The key mechanic" above), the
object `self.context` points at is simply never touched again once `database_connection`'s
own run ends — no snapshot/copy is needed, holding the reference is enough. Models do not
need this because a model lives and dies inside one verb run — a model *wants* to follow the
live context, read fresh at the point of use. Regression test:
`test_context.py::test_vector_db_keeps_the_context_of_its_own_run`.

### Distributed repair

`idist.Parallel` spawns fresh processes, which re-import modules — so the context is empty
in every child rank. `Train._training` already receives `results_dir`, so it repopulates
there, on the path every rank passes through:

```python
update_context(
    results_dir=results_dir,
    verb="train",
)
```

`update_context` rather than a fresh `run_context` so the single-process path (where
`_training` is called directly, inside the context `run()` already installed) merges into
that existing context rather than replacing it — so anything a model added during
`setup_model` survives. In a spawned rank it simply fills that process's empty context, which
needs no release since the process exits when training finishes.

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
        self.activations = []

    def train_batch(self, batch):
        ...
        self.activations.append(hidden.detach().cpu().numpy())
        return {"loss": loss.item()}

    def train_post_epoch(self):
        context = get_context()
        np.save(context["results_dir"] / "activations.npy",
                np.concatenate(self.activations))
        self.activations.clear()
```

**Call `get_context()` at the point of use, never stash it in `__init__`.** Distributed
training pickles the model into per-rank child processes before those processes have set up
their own context; a handle taken in the parent's `__init__` is a stale, disconnected copy
once unpickled in a child, so it never sees the `update_context` call the child makes for its
own rank. Reading `get_context()` fresh inside the hook always returns the live context for
whichever process is actually running it.

The natural flush points are the duck-typed zero-arg hooks `train_post_epoch`,
`validate_post_epoch` (both pre-existing) and `test_post_epoch` (added by this change to
round out the set, wired into `create_tester` in `pytorch_ignite.py`) — none of which had any
documentation before this change. Hyrax calls whichever of these a model defines once at the
end of the corresponding epoch/pass; none of them are supported under distributed training
(`world_size > 1` raises `NotImplementedError` if the model defines one).

Users may also add their own keys. The context is fresh at the start of each verb run and
released at the end, so user keys last for that run only.

## Cleanups included

1. **Deleted the dead context** in `infer.py`.
2. **Standardized `results_dir` on `Path`.** `save_to_database.py` was the only site passing
   a `str`. Safe: every ChromaDB use coerces via `str(results_dir)`, and
   `database_connection.py` already handed Qdrant a `Path`. Added a matching `str()`
   coercion in `qdrantdb_impl.py` so both backends are explicit. `model_exporters.py`
   uses the `/` operator and *requires* a `Path`.
3. **Renamed `ctx` → `context`** in `export_to_onnx` (before the parameter was dropped
   entirely).
4. **Dropped the context parameter** from `VectorDB.__init__`, `vector_db_factory` and
   `export_to_onnx`; they call `get_context()` instead. Replaced the vague *"an instance of
   the context object"* docstring with a real explanation, and fixed the stale
   `results_dir: str` annotations on the ChromaDB worker helpers, which have received a
   `Path` since the standardization above. Test call sites that used to construct a
   `VectorDB` with a hand-built context dict now do
   `with run_context("test", results_dir=tmp_path): ChromaDB(config)` instead
   (`test_chromadb_impl.py`, `test_qdrant_impl.py`).

Not in scope: the `#!` TODO in `qdrantdb_impl.py` (near the `vector_size` config read,
in `create()`) asking for a vector-size context key. `ContextKeys` makes it a clean follow-up.

## Files

**New:** `src/hyrax/context.py`, `tests/hyrax/test_context.py`, `specs/model_run_context.md`

**Modified:** `src/hyrax/__init__.py`, `src/hyrax/config_utils.py` (`create_results_dir` now
populates the context), `src/hyrax/model_exporters.py`, `src/hyrax/pytorch_ignite.py` (adds
the `test_post_epoch` hook), `src/hyrax/vector_dbs/{vector_db_interface,vector_db_factory,
qdrantdb_impl,chromadb_impl}.py`, verbs (`train`, `infer`, `test`, `infer_stream`, `to_onnx`,
`save_to_database`, `database_connection`, `verb_registry`), `tests/hyrax/{conftest,
test_chromadb_impl,test_qdrant_impl,test_test,test_train}.py`, `docs/model_class_reference.rst`,
`HYRAX_GUIDE.md`.

`create_splits.py`, `engine.py`, and `reduce_dimensions.py` needed no changes in the end —
`create_results_dir` wires their context automatically.

## Verification

`tests/hyrax/test_context.py` covers:

- `get_context()` returns the *same object* throughout one `run_context(...)` block.
- Each run gets its own context object: one captured from a finished run keeps that run's
  values even after a later run's context replaces what `get_context()` returns.
- `run_context` installs the documented keys, coercing `results_dir` to `Path` from both
  `str` and `Path`, and accepts extra keys (as `to_onnx` does for `ml_framework`).
- `run_context` releases its context back to empty both on normal exit and when the body
  raises.
- `update_context` preserves existing keys (so a model's `setup_model`-time additions
  survive) and coerces `results_dir` to `Path`.
- `use_context` re-installs a captured context for code that outlives its run.
- Missing keys raise a `KeyError` naming the available keys, with a distinct message when the
  context is empty outside any verb run.
- `clear_context` releases a context populated outside a verb run (e.g. by a test).
- Wrapping `run()` in `run_context` doesn't flatten a verb's real call signature — notebook
  users get accurate help text and completion.

Elsewhere:

- **End-to-end** (`test_train.py::test_model_writes_to_results_dir_during_train`,
  `test_test.py::test_model_writes_to_results_dir_during_test`, via the
  `context_writing_loopback` fixture in `conftest.py`): a model whose `train_post_epoch` /
  `test_post_epoch` calls `get_context()` and writes JSON into `results_dir`; running `h.train()`
  / `h.test()` lands the file in that run's own results dir, tagged with the right `verb`.
- **`test_context.py::test_vector_db_keeps_the_context_of_its_own_run`**: `save_to_database`
  then `database_connection` hand back a live database; a later `h.infer()` call installs and
  releases its own context; the held database still resolves queries against its own
  directory throughout.
- `tests/hyrax/conftest.py` gained an autouse `clean_context` fixture that clears the context
  before and after every test, so a context set directly (as the vector-db tests do) can't
  leak into whatever test runs next.

Full fast suite (`tests/`, per the `python -m pytest -m "not slow"` command in this repo's
`CLAUDE.md`): 629 passed, 10 deselected. One pre-existing collection error is unrelated
(`test_range_read_lsst_dataset.py` needs `fitsio`). Running from the repo root instead of
`tests/` also collects `docs/` (per `pyproject.toml`'s `testpaths`) and picks up stale
`prepare_inputs.py` copies left behind in `docs/pre_executed/**/results/` by
previously-executed notebooks — unrelated to this change.
