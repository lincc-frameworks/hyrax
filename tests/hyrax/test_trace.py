"""
Unit tests for the tracing subsystem in hyrax.trace.

These tests cover:
  1. TraceContext with a falsy trace arg is a no-op (config unchanged, no global trace set).
  2. TraceContext with a truthy trace arg sets the global trace_result and modifies config.
  3. TraceContext cleanup: global trace_result is cleared and DataProvider.__len__ is restored
     after __exit__ (normal and exceptional exit paths).
  4. trace_verb_data decorator: verb returns a TraceResult when trace= is supplied, or its
     ordinary return value when trace is not supplied.
"""

import pytest

# ---------------------------------------------------------------------------
# Minimal config helper
# ---------------------------------------------------------------------------


def _make_config(epochs=5, batch_size=10, use_cache=True):
    """Return a minimal config dict accepted by TraceContext."""
    return {
        "train": {"epochs": epochs},
        "data_loader": {"batch_size": batch_size},
        "data_set": {"use_cache": use_cache},
    }


# ---------------------------------------------------------------------------
# TraceContext – no-op when trace is falsy
# ---------------------------------------------------------------------------


def test_trace_context_noop_when_trace_is_none():
    """TraceContext with trace=None leaves config untouched and never sets global trace."""
    from hyrax.trace import TraceContext, get_trace

    config = _make_config()
    with TraceContext(None, config) as modified_config:
        assert get_trace() is None
        assert modified_config["train"]["epochs"] == 5
        assert modified_config["data_loader"]["batch_size"] == 10
        assert modified_config["data_set"]["use_cache"] is True

    # Still None after exit
    assert get_trace() is None


def test_trace_context_noop_when_trace_is_zero():
    """TraceContext with trace=0 (falsy int) leaves config untouched."""
    from hyrax.trace import TraceContext, get_trace

    config = _make_config()
    with TraceContext(0, config) as modified_config:
        assert get_trace() is None
        assert modified_config["train"]["epochs"] == 5

    assert get_trace() is None


# ---------------------------------------------------------------------------
# TraceContext – active trace
# ---------------------------------------------------------------------------


def test_trace_context_sets_global_trace_result():
    """TraceContext with trace=N sets get_trace() to a TraceResult inside the context."""
    from hyrax.trace import TraceContext, TraceResult, get_trace

    config = _make_config()
    with TraceContext(3, config):
        tr = get_trace()
        assert tr is not None
        assert isinstance(tr, TraceResult)

    assert get_trace() is None


def test_trace_context_modifies_config():
    """TraceContext applies expected config overrides while active."""
    from hyrax.trace import TraceContext

    config = _make_config(epochs=20, batch_size=64, use_cache=True)
    trace_batch_size = 4
    with TraceContext(trace_batch_size, config) as modified_config:
        assert modified_config["train"]["epochs"] == 1
        assert modified_config["data_loader"]["batch_size"] == trace_batch_size
        assert modified_config["data_set"]["use_cache"] is False


# ---------------------------------------------------------------------------
# TraceContext – DataProvider.__len__ shim is applied and removed
# ---------------------------------------------------------------------------


def test_trace_context_restores_dataprovider_len_after_exit():
    """
    During an active trace TraceContext shims DataProvider.__len__.
    After __exit__ the original method is restored.
    """
    from hyrax.data_sets.data_provider import DataProvider

    from hyrax.trace import TraceContext

    original_len = DataProvider.__dict__.get("__len__")

    config = _make_config()
    with TraceContext(3, config):
        # The shim should replace the original __len__
        assert DataProvider.__dict__.get("__len__") is not original_len

    # After the context the original should be back
    assert DataProvider.__dict__.get("__len__") is original_len


def test_trace_context_restores_dataprovider_len_on_exception():
    """
    TraceContext restores DataProvider.__len__ even when an exception is raised
    inside the context block.
    """
    from hyrax.data_sets.data_provider import DataProvider

    from hyrax.trace import TraceContext, get_trace

    original_len = DataProvider.__dict__.get("__len__")

    with pytest.raises(RuntimeError, match="test error"):
        with TraceContext(3, _make_config()):
            raise RuntimeError("test error")

    # Cleanup must have run despite the exception
    assert get_trace() is None
    assert DataProvider.__dict__.get("__len__") is original_len


# ---------------------------------------------------------------------------
# trace_verb_data decorator
# ---------------------------------------------------------------------------


def test_trace_verb_data_returns_trace_result_when_trace_set(loopback_hyrax):
    """
    The @trace_verb_data decorator causes a verb to return a TraceResult instead of
    its normal return value when trace=<N> is supplied.
    """
    from hyrax.trace import TraceResult

    h, _ = loopback_hyrax
    result = h.infer(trace=3)
    assert isinstance(result, TraceResult)


def test_trace_verb_data_returns_normal_value_without_trace(loopback_hyrax):
    """
    When trace is not supplied the verb returns its ordinary value (not a TraceResult).
    """
    from hyrax.trace import TraceResult

    h, _ = loopback_hyrax
    result = h.infer()
    assert not isinstance(result, TraceResult)


# ---------------------------------------------------------------------------
# TraceResult stage capture
# ---------------------------------------------------------------------------


def test_trace_result_populates_resolve_data_stage(loopback_hyrax):
    """
    Running infer(trace=N) produces a TraceResult whose 'resolve_data' stage
    contains at least one captured call from the DataProvider pipeline.
    """
    from hyrax.trace import TraceResult, TraceStage

    h, _ = loopback_hyrax
    trace_result = h.infer(trace=3)

    assert isinstance(trace_result, TraceResult)
    stage = trace_result["resolve_data"]
    assert isinstance(stage, TraceStage)
    assert len(stage) > 0, "resolve_data stage should have at least one captured call"

    # Each call should be printable
    first_call = stage[0]
    assert str(first_call)


def test_trace_result_populates_collate_stage(loopback_hyrax):
    """
    Running infer(trace=N) produces a TraceResult whose 'collate' stage
    contains at least one captured call from the DataProvider pipeline.
    """
    from hyrax.trace import TraceResult, TraceStage

    h, _ = loopback_hyrax
    trace_result = h.infer(trace=3)

    assert isinstance(trace_result, TraceResult)
    stage = trace_result["collate"]
    assert isinstance(stage, TraceStage)
    assert len(stage) > 0, "collate stage should have at least one captured call"
