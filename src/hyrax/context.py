"""Run-scoped context shared by verbs, models, vector databases, and exporters.

A Hyrax verb run produces a results directory and a handful of other facts about
the run (which verb, which distributed rank). Several pieces of Hyrax need those
facts but are not handed them directly: models are constructed with only
``config`` and ``data_sample``, and vector databases and the ONNX exporter
receive them through an informal ``context`` dictionary.

This module holds one context for the process. Verbs populate it with
:func:`init_context`; everything else reads it with :func:`get_context`.

Typical usage in a model::

    from hyrax import get_context

    @hyrax_model
    class MyModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.context = get_context()

        def train_post_epoch(self):
            rank = self.context["rank"]
            np.save(self.context["results_dir"] / f"acts_rank{rank}.npy", self.acts)

The dictionary returned by :func:`get_context` is the same object for the life
of the process -- :func:`init_context` clears and repopulates it in place rather
than replacing it. That means client code never has to think about
initialization order: a handle taken before a verb populates the context still
sees the values once they arrive.
"""

import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

__all__ = [
    "ContextKeys",
    "RunContext",
    "clear_context",
    "get_context",
    "init_context",
    "update_context",
]


class ContextKeys(TypedDict, total=False):
    """The keys Hyrax puts in the run context.

    This is a documentation and type-annotation aid; the context is an ordinary
    dictionary and is not validated against this definition.

    Attributes
    ----------
    results_dir : Path
        The results directory for the current run.
    verb : str
        The name of the verb that started the run, e.g. ``"train"``, ``"infer"``,
        ``"test"``, ``"infer_stream"``, ``"onnx"``, ``"vector-db"``.
    rank : int
        The distributed rank of the current process. ``0`` when not running
        distributed.
    world_size : int
        The number of distributed processes. ``1`` when not running distributed.
    ml_framework : str
        The source framework for a model export. Only set by the ``to_onnx`` verb.
    """

    results_dir: Path
    verb: str
    rank: int
    world_size: int
    ml_framework: str


class RunContext(dict):
    """A plain ``dict`` that explains itself when a key is missing.

    Behaves exactly like a ``dict`` in every other respect, so it can be passed
    anywhere a context dictionary is expected.
    """

    def __missing__(self, key):
        msg = f"'{key}' is not in the Hyrax run context. "
        msg += f"Available keys: {sorted(self)}. "
        msg += "The run context is populated by a Hyrax verb (train, infer, test, ...) "
        msg += "and is empty when a model is constructed outside of a verb run."
        raise KeyError(msg)


# One object for the life of the process. Never rebound - only cleared and updated
# in place - so handles taken before init_context() stay valid.
_context = RunContext()


def get_context() -> RunContext:
    """Get the run context for this process.

    Safe to call at any time. Before a verb has started a run the context is
    empty, and reading a key from it raises a ``KeyError`` explaining why.

    Returns
    -------
    RunContext
        The process-wide run context. This is always the same object, so a
        handle taken now will reflect values a verb adds later.
    """
    return _context


def init_context(results_dir, verb: str, **extra) -> RunContext:
    """Start a new run context, discarding any previous run's values.

    This should be called by code that controls overall hyrax execution, e.g. a
    Verb's ``run()`` method, immediately after creating its results directory.

    Parameters
    ----------
    results_dir : Path or str
        The results directory for this run. Coerced to ``Path``.
    verb : str
        The name of the verb starting the run.
    **extra
        Any additional keys to place in the context.

    Returns
    -------
    RunContext
        The run context, for convenience. This is the same object
        :func:`get_context` returns.
    """
    _context.clear()
    return update_context(results_dir=Path(results_dir), verb=verb, **extra)


def update_context(**kwargs) -> RunContext:
    """Merge values into the run context, leaving existing keys in place.

    Use this rather than :func:`init_context` when adding to a run that is
    already underway, so that anything a model has stashed in the context is
    preserved.

    If ``results_dir`` is given it is coerced to ``Path``. If ``rank`` and
    ``world_size`` are not given they are filled in from ignite's distributed
    configuration.

    Returns
    -------
    RunContext
        The run context, for convenience.
    """
    # Imported here rather than at module scope to keep this module cheap for
    # user model code to import.
    import ignite.distributed as idist

    if "results_dir" in kwargs:
        kwargs["results_dir"] = Path(kwargs["results_dir"])

    kwargs.setdefault("rank", idist.get_rank())
    kwargs.setdefault("world_size", idist.get_world_size())

    _context.update(kwargs)
    return _context


def clear_context() -> None:
    """Empty the run context.

    Valid to call whether or not a run is active.
    """
    _context.clear()
