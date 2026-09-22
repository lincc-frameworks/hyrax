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


class _ImageStream(KafkaStreamDataset):
    """KafkaStreamDataset subclass exposing the `get_<field>` accessors the provider needs.

    ``StreamingDataProvider`` reads each requested field through ``get_<field>(sample)`` on
    the wrapped dataset, so a stream must define one getter per field it offers.
    """

    def get_image(self, sample):
        """Return the image payload of one decoded sample."""
        return sample["image"]

    def get_object_id(self, sample):
        """Return the object id of one decoded sample."""
        return str(sample["object_id"])


def _msg(object_id, image):
    return FakeMessage(json.dumps({"object_id": object_id, "image": image}))


def _build_stream_config(tmp_path, batch_size=2):
    """Configure a Hyrax instance to run infer_stream over a Kafka-backed data request."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["data_loader"]["batch_size"] = batch_size

    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "test-topic"
    ds_config["batch_flush_timeout"] = 0.0  # flush partial batches on the first empty poll

    h.config["data_request"] = {
        "infer_stream": {
            "data": {
                "dataset_class": "_ImageStream",
                "data_location": "./",
                "primary_id_field": "object_id",
                "fields": ["image"],
            }
        }
    }

    weights = tmp_path / "weights.pth"
    weights.write_text("")  # HyraxLoopback.load is a no-op; contents are irrelevant
    h.config["infer_stream"]["model_weights_file"] = str(weights)
    return h


def _record_consumers(monkeypatch, messages):
    """Patch consumer creation and return the list of FakeConsumers as they are built.

    Each stream gets its own copy of ``messages`` so a second session sees the same data
    the first did, and every consumer is recorded so tests can assert on teardown.
    """
    created = []

    def make(self):
        consumer = FakeConsumer(list(messages), on_exhausted=self.stop)
        created.append(consumer)
        return consumer

    monkeypatch.setattr(KafkaStreamDataset, "_make_consumer", make)
    return created


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
    ds_config["topics"] = "test-topic"
    ds_config["batch_flush_timeout"] = 0.0  # flush partial batches on the first empty poll

    h.config["data_request"] = {
        "infer_stream": {
            "data": {
                "dataset_class": "_ImageStream",
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


def test_close_releases_consumer_while_iteration_is_still_referenced(tmp_path, monkeypatch):
    """close() releases the consumer even though the suspended iterator is still alive.

    A suspended generator does eventually run its own ``finally`` -- but only once nothing
    references it. Refcounting usually arranges that promptly in a script, and does not in
    a notebook, where the retained traceback pins the frames indefinitely. Holding the
    iterator here stands in for that, so what this pins down is deterministic teardown at
    close() rather than cleanup that happens to arrive later on its own.
    """
    h = _build_stream_config(tmp_path)
    created = _record_consumers(monkeypatch, [_msg(f"id{i}", [[float(i)]]) for i in range(3)])

    session = h.infer_stream()
    iterator = iter(session.data_loader)
    next(iterator)  # a batch is in flight, so the stream generator is suspended

    assert created, "no Kafka consumer was created"
    assert not any(consumer.closed for consumer in created)

    session.close()

    assert all(consumer.closed for consumer in created)
    # Teardown reaches close() from several directions; it must land exactly once.
    assert all(consumer.close_count == 1 for consumer in created)
    assert iterator is not None  # still referenced: nothing was collected behind our back


def test_session_close_releases_consumer_when_body_raises(tmp_path, monkeypatch):
    """An exception inside the `with` body still drops the Kafka connection.

    Mirrors how callers actually drive the session (iterating ``session.data_loader``, as
    hyrax_alerts does) and keeps both the iterator and the traceback alive afterwards, the
    way a notebook does, so teardown has to come from ``__exit__`` rather than from
    collection.
    """
    h = _build_stream_config(tmp_path)
    created = _record_consumers(monkeypatch, [_msg(f"id{i}", [[float(i)]]) for i in range(3)])

    held = []
    try:
        with h.infer_stream() as session:
            iterator = iter(session.data_loader)
            held.append(iterator)
            for _batch in iterator:
                raise ValueError("pipeline blew up")
    except ValueError as err:
        held.append(err.__traceback__)

    assert len(held) == 2, "the pipeline error did not propagate"
    assert created, "no Kafka consumer was created"
    assert all(consumer.closed for consumer in created)
    assert all(consumer.close_count == 1 for consumer in created)


def test_stream_restarts_after_a_failed_session(tmp_path, monkeypatch):
    """A new session in the same interpreter works after an earlier one failed.

    The fakes cannot reproduce a broker-side rebalance, so what this pins down is the
    precondition for recovery: the failed run leaves nothing open, and the next run builds
    a fresh consumer and reads the topic normally.
    """
    messages = [_msg(f"id{i}", [[float(i)]]) for i in range(3)]
    created = _record_consumers(monkeypatch, messages)

    failed = _build_stream_config(tmp_path / "run1")
    with pytest.raises(ValueError):
        with failed.infer_stream() as session:
            for _batch, _results in session:
                raise ValueError("pipeline blew up")

    assert all(consumer.closed for consumer in created)
    closed_after_first_run = len(created)

    retry = _build_stream_config(tmp_path / "run2")
    seen_ids = []
    with retry.infer_stream() as session:
        for batch, _results in session:
            seen_ids.extend(batch["object_id"])

    assert sorted(seen_ids) == ["id0", "id1", "id2"]
    assert len(created) > closed_after_first_run, "the retry reused the failed run's consumer"
    assert all(consumer.closed for consumer in created)


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
