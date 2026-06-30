"""Tests for KafkaStreamDataset's latency-bounded batching and flat decoding.

A FakeConsumer stands in for confluent_kafka.Consumer so the tests never touch a real
broker. Its ``poll`` returns queued messages and then ``None`` (an empty poll); an
optional ``on_exhausted`` hook lets a test stop the stream once its messages run out.

The stream is a dumb decoder: it yields ``list[dict]`` batches of *flat* JSON dicts.
Structuring/collation is covered in ``test_streaming_data_provider.py``.
"""

import json

import pytest

import hyrax
from hyrax.datasets.kafka_stream_dataset import KafkaStreamDataset


class FakeMessage:
    """Minimal stand-in for a confluent_kafka Message."""

    def __init__(self, value, error=None):
        self._value = value
        self._error = error

    def value(self):
        """Return the message payload."""
        return self._value

    def error(self):
        """Return the message error, or None for a normal message."""
        return self._error


class FakeConsumer:
    """Returns queued messages, then None (empty poll) on every subsequent call."""

    def __init__(self, messages, on_exhausted=None):
        self._messages = list(messages)
        self._on_exhausted = on_exhausted
        self.closed = False

    def poll(self, timeout):
        """Pop the next queued message, or None once the queue is empty."""
        if self._messages:
            return self._messages.pop(0)
        if self._on_exhausted is not None:
            self._on_exhausted()
        return None

    def consume(self, num_messages, timeout):
        """Drain up to num_messages queued messages; signal exhaustion when empty."""
        drained = []
        while self._messages and len(drained) < num_messages:
            drained.append(self._messages.pop(0))
        if not drained and self._on_exhausted is not None:
            self._on_exhausted()
        return drained

    def subscribe(self, topics):
        """No-op subscribe to match the confluent_kafka Consumer interface."""
        pass

    def close(self):
        """Record that the consumer was closed."""
        self.closed = True


def _make_message(object_id, image):
    return FakeMessage(json.dumps({"object_id": object_id, "image": image}))


def _build_dataset(batch_size=5, batch_flush_timeout=100.0):
    """Construct a KafkaStreamDataset with a configured topic and batch settings."""
    h = hyrax.Hyrax()
    h.config["data_loader"]["batch_size"] = batch_size
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topic"] = "test-topic"
    ds_config["batch_flush_timeout"] = batch_flush_timeout
    return KafkaStreamDataset(h.config)


def _patch_consumer(monkeypatch, dataset, messages, stop_when_exhausted=False):
    """Make the dataset use a single FakeConsumer over ``messages``."""
    on_exhausted = dataset.stop if stop_when_exhausted else None
    consumer = FakeConsumer(messages, on_exhausted=on_exhausted)
    monkeypatch.setattr(dataset, "_make_consumer", lambda: consumer)
    return consumer


def test_missing_topic_raises():
    """A topic must be configured; the TOML `false` sentinel is rejected."""
    h = hyrax.Hyrax()
    h.config["data_set"]["KafkaStreamDataset"]["topic"] = False
    with pytest.raises(ValueError, match="topic"):
        KafkaStreamDataset(h.config)


def test_len_raises():
    """A live stream has no length."""
    dataset = _build_dataset()
    with pytest.raises(TypeError, match="no length"):
        len(dataset)


def test_decode_returns_flat_dict():
    """_decode returns the parsed JSON object unchanged (no structuring)."""
    dataset = _build_dataset()
    msg = FakeMessage(json.dumps({"object_id": "x", "image": [[1.0]], "flux": 2.0}))

    sample = dataset._decode(msg)

    assert sample == {"object_id": "x", "image": [[1.0]], "flux": 2.0}


def test_full_batch_emitted_at_batch_size(monkeypatch):
    """When batch_size messages arrive, a full batch is yielded immediately."""
    dataset = _build_dataset(batch_size=3)
    messages = [_make_message(f"id{i}", [[float(i), 0.0]]) for i in range(3)]
    _patch_consumer(monkeypatch, dataset, messages, stop_when_exhausted=True)

    batches = list(dataset)

    assert len(batches) == 1
    assert len(batches[0]) == 3
    assert [s["object_id"] for s in batches[0]] == ["id0", "id1", "id2"]


def test_partial_batch_flushed_on_timeout(monkeypatch):
    """Fewer than batch_size messages are flushed once the wait elapses."""
    # batch_flush_timeout=0 makes the deadline elapse on the first empty poll.
    dataset = _build_dataset(batch_size=10, batch_flush_timeout=0.0)
    messages = [_make_message("a", [[1.0]]), _make_message("b", [[2.0]])]
    _patch_consumer(monkeypatch, dataset, messages, stop_when_exhausted=True)

    batches = list(dataset)

    assert len(batches) == 1
    assert len(batches[0]) == 2  # short batch, not the requested 10


def test_stop_flushes_remaining(monkeypatch):
    """stop() flushes whatever has accumulated before iteration ends."""
    # Large flush timeout: the only reason a batch is emitted is the stop() flush.
    dataset = _build_dataset(batch_size=5, batch_flush_timeout=100.0)
    messages = [_make_message("a", [[1.0]]), _make_message("b", [[2.0]])]
    _patch_consumer(monkeypatch, dataset, messages, stop_when_exhausted=True)

    batches = list(dataset)

    assert len(batches) == 1
    assert len(batches[0]) == 2


def test_consumer_closed_after_iteration(monkeypatch):
    """The Kafka consumer is closed when iteration ends."""
    dataset = _build_dataset(batch_size=2)
    messages = [_make_message("a", [[1.0]]), _make_message("b", [[2.0]])]
    consumer = _patch_consumer(monkeypatch, dataset, messages, stop_when_exhausted=True)

    list(dataset)

    assert consumer.closed


def test_peek_sample_buffers_and_replays(monkeypatch):
    """peek_sample returns a flat sample that is not lost; it leads the first batch."""
    dataset = _build_dataset(batch_size=5, batch_flush_timeout=0.0)
    messages = [_make_message("first", [[1.0]]), _make_message("second", [[2.0]])]
    _patch_consumer(monkeypatch, dataset, messages, stop_when_exhausted=True)

    peeked = dataset.peek_sample()
    assert peeked["object_id"] == "first"

    batches = list(dataset)

    assert len(batches) == 1
    # The peeked message is replayed as the first sample of the first batch.
    assert [s["object_id"] for s in batches[0]] == ["first", "second"]
