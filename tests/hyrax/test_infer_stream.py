"""Tests for the infer_stream verb's data-source-driven session.

Uses the FakeConsumer from ``test_kafka_stream_dataset.py`` (patched onto
``KafkaStreamDataset`` at the class level, since ``infer_stream`` builds the dataset
internally) and the trivial ``HyraxLoopback`` model so no real broker or weights are
needed.
"""

import json

import pytest
from test_kafka_stream_dataset import FakeConsumer, FakeMessage  # noqa: I001

import hyrax
from hyrax.datasets.kafka_stream_dataset import KafkaStreamDataset


def _msg(object_id, image):
    return FakeMessage(json.dumps({"object_id": object_id, "image": image}))


def test_infer_stream_requires_sample_or_request():
    """With no sample_batch and no data_request, run() raises a clear error."""
    h = hyrax.Hyrax()
    with pytest.raises(ValueError, match="sample_batch"):
        h.infer_stream()


def test_infer_stream_iterates_streaming_dataset(tmp_path, monkeypatch):
    """A configured [data_request.infer_stream] yields (batch, results) on iteration."""
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["data_loader"]["batch_size"] = 2

    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topic"] = "test-topic"
    ds_config["batch_flush_timeout"] = 0.0  # flush partial batches on the first empty poll

    h.config["data_request"] = {
        "infer_stream": {
            "data": {
                "dataset_class": "KafkaStreamDataset",
                "data_location": "./",
                "primary_id_field": "object_id",
                "fields": ["image"],
            }
        }
    }

    weights = tmp_path / "weights.pth"
    weights.write_text("")  # HyraxLoopback.load is a no-op; contents are irrelevant
    h.config["infer_stream"]["model_weights_file"] = str(weights)

    # One FakeConsumer per dataset instance; on exhaustion it stops that stream so the
    # iteration terminates. _make_consumer receives `self` (the stream) when patched.
    messages = [_msg(f"id{i}", [[float(i)]]) for i in range(3)]
    monkeypatch.setattr(
        KafkaStreamDataset,
        "_make_consumer",
        lambda self: FakeConsumer(messages, on_exhausted=self.stop),
    )

    seen_ids = []
    with h.infer_stream() as session:
        for batch, results in session:
            ids = list(batch["object_id"])
            seen_ids.extend(ids)
            # Loopback returns its image input; one result row per object.
            assert results.shape[0] == len(ids)

    # The peeked sample (used for model pre-flighting) is not lost.
    assert sorted(seen_ids) == ["id0", "id1", "id2"]


def test_session_without_source_is_not_iterable():
    """A session built without a data_loader (manual path) cannot be iterated."""
    from hyrax.verbs.infer_stream import InferStreamSession

    session = InferStreamSession(
        process_func=lambda *args: None,
        save_batch_callback=lambda *args: None,
        config={"infer_stream": {"save_model_output": False}},
        results_dir=None,
        close_logger_fn=lambda: None,
        load_dataset_fn=lambda *args: None,
        data_loader=None,
        provider=None,
    )
    with pytest.raises(RuntimeError, match="no data source"):
        list(session)
