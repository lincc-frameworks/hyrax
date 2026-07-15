"""Tests for the train_stream verb's streaming training session.

Reuses the FakeConsumer/FakeMessage stand-ins from ``test_kafka_stream_dataset.py``
(patched onto ``KafkaStreamDataset`` at the class level, since ``train_stream`` builds
the dataset internally) and the trivial ``HyraxLoopback`` model, so no real broker or
weights are needed. The manual-path tests construct ``TrainStreamSession`` directly with a
spy ``process_func`` to exercise the empty / small-batch skipping logic in isolation.
"""

import json
from pathlib import Path

import pytest
from test_kafka_stream_dataset import FakeConsumer, FakeMessage  # noqa: I001

import hyrax
from hyrax.datasets.kafka_stream_dataset import KafkaStreamDataset
from hyrax.verbs.train_stream import TrainStreamSession


def _msg(object_id, image):
    return FakeMessage(json.dumps({"object_id": object_id, "image": image}))


def _manual_session(tmp_path, process_func, *, min_batch_size=False, save_weights_every=False):
    """Build a TrainStreamSession directly (manual path) with a spy model and process_func."""

    class _SpyModel:
        def __init__(self):
            self.saved = []

        def save(self, path):
            self.saved.append(path)

    config = {
        "train_stream": {
            "weights_filename": "weights.pth",
            "save_weights_every": save_weights_every,
            "min_batch_size": min_batch_size,
        }
    }
    session = TrainStreamSession(
        process_func=process_func,
        model=_SpyModel(),
        config=config,
        results_dir=Path(tmp_path),
        close_logger_fn=lambda: None,
        data_loader=None,
        provider=None,
    )
    return session


def test_train_stream_run_cli_not_implemented():
    """train_stream is programmatic only; the CLI entrypoint raises."""
    h = hyrax.Hyrax()
    verb = hyrax.verbs.fetch_verb_class("train_stream")(h.config)
    with pytest.raises(NotImplementedError, match="programmatic API"):
        verb.run_cli()


def test_train_stream_requires_sample_or_request():
    """With no sample_batch and no data_request, run() raises a clear error."""
    h = hyrax.Hyrax()
    with pytest.raises(ValueError, match="sample_batch"):
        h.train_stream()


def test_train_stream_iterates_streaming_dataset(tmp_path, monkeypatch):
    """A configured [data_request.train_stream] yields (batch, metrics) and saves weights."""
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["data_loader"]["batch_size"] = 2

    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "test-topic"
    ds_config["batch_flush_timeout"] = 0.0  # flush partial batches on the first empty poll

    h.config["data_request"] = {
        "train_stream": {
            "data": {
                "dataset_class": "KafkaStreamDataset",
                "data_location": "./",
                "primary_id_field": "object_id",
                "fields": ["image"],
            }
        }
    }

    messages = [_msg(f"id{i}", [[float(i)]]) for i in range(3)]
    monkeypatch.setattr(
        KafkaStreamDataset,
        "_make_consumer",
        lambda self: FakeConsumer(messages, on_exhausted=self.stop),
    )

    seen_ids = []
    with h.train_stream() as session:
        for batch, metrics in session:
            seen_ids.extend(list(batch["object_id"]))
            # HyraxLoopback.train_batch is a no-op that returns a loss dict per batch.
            assert "loss" in metrics
        results_dir = session._results_dir

    # The peeked sample (used for model pre-flighting) is not lost.
    assert sorted(seen_ids) == ["id0", "id1", "id2"]
    # Final weights were persisted on close.
    assert (results_dir / "example_model.pth").exists()


def test_train_stream_close_returns_model_and_is_idempotent(tmp_path):
    """close() returns the model and repeated calls are safe (no double-save error)."""
    calls = []

    def process_func(engine, batch):
        calls.append(batch)
        return {"loss": 1.0}

    session = _manual_session(tmp_path, process_func)
    model = session._model
    assert session.close() is model
    # Idempotent: second close returns the same model and does not raise.
    assert session.close() is model
    # save_weights was invoked on close.
    assert model.saved


def test_process_after_close_raises(tmp_path):
    """Calling process()/train_batch() after close() raises RuntimeError."""
    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0})
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.process({"object_id": ["a", "b"]})


def test_empty_batch_is_skipped(tmp_path):
    """An empty batch is skipped: no training step, returns None."""
    calls = []

    def process_func(engine, batch):
        calls.append(batch)
        return {"loss": 1.0}

    session = _manual_session(tmp_path, process_func)
    assert session.process({"object_id": []}) is None
    assert calls == []


def test_min_batch_size_skips_small_batches(tmp_path):
    """Batches smaller than min_batch_size are skipped; larger ones are trained."""
    calls = []

    def process_func(engine, batch):
        calls.append(batch)
        return {"loss": 1.0}

    session = _manual_session(tmp_path, process_func, min_batch_size=2)

    # One-sample batch is below the threshold -> skipped.
    assert session.process({"object_id": ["a"]}) is None
    assert calls == []

    # Two-sample batch meets the threshold -> trained.
    result = session.process({"object_id": ["a", "b"]})
    assert result == {"loss": 1.0}
    assert len(calls) == 1


def test_save_weights_every(tmp_path):
    """Weights are checkpointed every N processed batches."""
    session = _manual_session(
        tmp_path,
        lambda engine, batch: {"loss": 1.0},
        save_weights_every=2,
    )
    model = session._model

    session.process({"object_id": ["a", "b"]})  # batch 1, no save
    assert model.saved == []
    session.process({"object_id": ["c", "d"]})  # batch 2, save
    assert len(model.saved) == 1


def test_session_without_source_is_not_iterable(tmp_path):
    """A session built without a data_loader (manual path) cannot be iterated."""
    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0})
    with pytest.raises(RuntimeError, match="no data source"):
        list(session)
