"""Run-scoped context shared by verbs, models, vector databases, and exporters.

A Hyrax verb run produces a results directory and a handful of other facts about
the run (which verb, which distributed rank). Several pieces of Hyrax need those
facts but are not handed them directly.

Each verb run gets its own context object. The ``Verb`` base class installs one
for the duration of ``run()`` and releases it when the run ends; everything else
reads the current one with :func:`get_context`.

Typical usage in a model::

    from hyrax import get_context

    @hyrax_model
    class MyModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config

        def train_post_epoch(self):
            context = get_context()
            rank = context["rank"]
            np.save(context["results_dir"] / f"acts_rank{rank}.npy", self.acts)

Because contexts are per-run objects that are never reused, an object that
outlives the run that created it can simply hold on to the context it was given
rather than copying values out of it. :class:`hyrax.vector_dbs.VectorDB` does
exactly that, so a database handed back by ``database_connection`` keeps pointing
at its own directory no matter what runs afterwards.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

__all__ = [
    "ContextKeys",
    "RunContext",
    "clear_context",
    "get_context",
    "run_context",
    "update_context",
    "use_context",
]


class ContextKeys(TypedDict, total=False):
    """The keys Hyrax puts in the run context.

    This is a documentation and type-annotation aid; the context is an ordinary
    dictionary and is not validated against this definition.

    Note: total=False means that all keys are optional so that type checkers
    don't complain about the context being empty before a verb has started a run.
    """

    results_dir: Path
    """The results directory for the current run."""

    verb: str
    """The name of the verb that started the run, e.g. ``"train"``, ``"infer"``"""

    ml_framework: str
    """The source framework for a model export. Only set by the ``to_onnx`` verb."""


class RunContext(dict):
    """A plain ``dict`` that explains itself when a key is missing.

    Behaves exactly like a ``dict`` in every other respect, so it can be passed
    anywhere a context dictionary is expected.
    """

    def __missing__(self, key):
        msg = f"'{key}' is not in the Hyrax run context. "
        msg += f"Available keys: {sorted(self)}. "
        msg += "The run context is populated by a Hyrax verb (train, infer, test, ...). "
        msg += "It is empty outside of a verb run (e.g. when constructing a model or vector DB directly)."
        raise KeyError(msg)


# The context for the run currently underway. Empty outside a run, so that
# get_context() is always safe to call.
_current: RunContext = RunContext()


def get_context() -> RunContext:
    """Get the run context for the run currently underway.

    Safe to call at any time. Outside a verb run the context is empty, and
    reading a key from it raises a ``KeyError`` explaining why.

    Returns
    -------
    RunContext
        The current run context. Each verb run gets a fresh object, so code that
        needs a context to outlive its run can hold this rather than copying
        values out of it.
    """
    return _current


@contextmanager
def run_context(verb: str, **extra) -> Generator[RunContext]:
    """Install a fresh context for one verb run, releasing it when the run ends.

    The ``Verb`` base class wraps every verb's ``run()`` in this, so verbs do not
    call it themselves; they fill in their results directory with
    :func:`update_context` once they have created it.

    Verb runs are not expected to nest. The context is released to empty rather
    than to whatever was current beforehand, so a verb that calls another verb
    is left without a context once the inner call returns.

    Parameters
    ----------
    verb : str
        The name of the verb starting the run.
    **extra
        Any additional keys to place in the context.

    Yields
    ------
    RunContext
        The newly installed context.
    """
    global _current

    _current = RunContext(verb=verb)
    try:
        update_context(**extra)
        yield _current
    finally:
        clear_context()


@contextmanager
def use_context(context: RunContext) -> Generator[RunContext]:
    """Re-enter a context captured earlier, for work that outlives its verb run.

    An object that keeps running model code after the verb that built it has
    returned - :class:`hyrax.verbs.infer_stream.InferStreamSession`, for example -
    captures ``get_context()`` at construction and re-installs it around that
    later work, so models see the run they belong to rather than an empty context.

    Parameters
    ----------
    context : RunContext
        A context captured earlier with :func:`get_context`.

    Yields
    ------
    RunContext
        The re-installed context.
    """
    global _current

    _current = context
    try:
        yield _current
    finally:
        clear_context()


def update_context(**kwargs) -> RunContext:
    """Merge values into the current run context, leaving existing keys in place.

    Verbs use this to record their results directory once they have created it.
    Existing keys are preserved, so anything a model has stashed in the context
    survives.

    If ``results_dir`` is given it is coerced to ``Path``. If ``rank`` and
    ``world_size`` are not given they are filled in from ignite's distributed
    configuration.

    Returns
    -------
    RunContext
        The current run context, for convenience.
    """
    if "results_dir" in kwargs:
        kwargs["results_dir"] = Path(kwargs["results_dir"])

    _current.update(kwargs)
    return _current


def clear_context() -> None:
    """Release the current run context.

    Verbs release their own context when they finish, so this is only needed to
    clean up after code that populated a context outside a verb run, such as a
    test.
    """
    global _current

    _current = RunContext()
