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

    def __init__(self, messages, on_exhausted=None, poll_error=None):
        self._messages = list(messages)
        self._on_exhausted = on_exhausted
        self._poll_error = poll_error
        self.closed = False
        self.close_count = 0
        self.commits = []

    def poll(self, timeout):
        """Pop the next queued message, or None once the queue is empty."""
        if self._poll_error is not None:
            raise self._poll_error
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

    def subscribe(self, topics, on_assign=None, on_revoke=None):
        """No-op subscribe to match the confluent_kafka Consumer interface."""
        pass

    def assignment(self):
        """No partitions assigned; enough for the diagnostics to render."""
        return []

    def commit(self, asynchronous=True):
        """Record a commit so tests can assert *when* offsets are advanced."""
        self.commits.append(asynchronous)

    def close(self):
        """Record that the consumer was closed, and catch double closes.

        confluent_kafka raises on a second close, so this does too: the stream is closed
        from several places (teardown, an abandoned generator's ``finally``) and a
        regression there must fail loudly rather than pass unnoticed.
        """
        self.close_count += 1
        if self.closed:
            raise RuntimeError("Consumer closed")
        self.closed = True


def _make_message(object_id, image):
    return FakeMessage(json.dumps({"object_id": object_id, "image": image}))


def _build_dataset(batch_size=5, batch_flush_timeout=100.0):
    """Construct a KafkaStreamDataset with a configured topic and batch settings."""
    h = hyrax.Hyrax()
    h.config["data_loader"]["batch_size"] = batch_size
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "test-topic"
    ds_config["batch_flush_timeout"] = batch_flush_timeout
    return KafkaStreamDataset(h.config)


def _patch_consumer(monkeypatch, dataset, messages, stop_when_exhausted=False, poll_error=None):
    """Make the dataset use a single FakeConsumer over ``messages``."""
    on_exhausted = dataset.stop if stop_when_exhausted else None
    consumer = FakeConsumer(messages, on_exhausted=on_exhausted, poll_error=poll_error)
    monkeypatch.setattr(dataset, "_make_consumer", lambda: consumer)
    return consumer


def test_missing_topic_raises():
    """A topic must be configured; the TOML `false` sentinel is rejected."""
    h = hyrax.Hyrax()
    h.config["data_set"]["KafkaStreamDataset"]["topics"] = False
    with pytest.raises(ValueError, match="topics"):
        KafkaStreamDataset(h.config)

    h.config["data_set"]["KafkaStreamDataset"]["topics"] = []
    with pytest.raises(ValueError, match="topics"):
        KafkaStreamDataset(h.config)


def test_data_location_overrides_bootstrap_and_topics_from_config():
    """Inline kafka:// URI takes precedence over configured bootstrap/topics."""
    h = hyrax.Hyrax()
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["bootstrap_servers"] = "config-broker:9092"
    ds_config["topics"] = ["config-topic"]

    dataset = KafkaStreamDataset(h.config, data_location="kafka://inline-broker:19092/inline-topic")

    assert dataset.bootstrap_servers == "inline-broker:19092"
    assert dataset.topics == ["inline-topic"]


def test_data_location_without_topic_uses_config_topics():
    """Inline broker override keeps configured topics when URI omits /topic."""
    h = hyrax.Hyrax()
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["bootstrap_servers"] = "config-broker:9092"
    ds_config["topics"] = ["config-topic-a", "config-topic-b"]

    dataset = KafkaStreamDataset(h.config, data_location="kafka://inline-broker:19092")

    assert dataset.bootstrap_servers == "inline-broker:19092"
    assert dataset.topics == ["config-topic-a", "config-topic-b"]


def test_topics_accepts_single_string_from_config():
    """A single configured topic string is normalized to a one-item list."""
    h = hyrax.Hyrax()
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "single-topic"

    dataset = KafkaStreamDataset(h.config)

    assert dataset.topics == ["single-topic"]


def test_topics_accepts_list_from_config():
    """A configured list of topics is used as-is."""
    h = hyrax.Hyrax()
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = ["topic-1", "topic-2"]

    dataset = KafkaStreamDataset(h.config)

    assert dataset.topics == ["topic-1", "topic-2"]


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
    assert consumer.close_count == 1
    assert dataset._consumer is None


def test_close_is_idempotent(monkeypatch):
    """Repeated close() calls close the underlying consumer exactly once.

    The stream is closed from several places -- session teardown, and an abandoned
    generator's ``finally`` whenever it is finally collected -- so the second call has to
    be a no-op.
    """
    dataset = _build_dataset()
    consumer = _patch_consumer(monkeypatch, dataset, [])
    dataset._ensure_consumer()

    dataset.close()
    dataset.close()

    assert consumer.close_count == 1
    assert dataset._consumer is None


def test_close_does_not_commit(monkeypatch):
    """close() must not advance offsets: it is reached on failure paths too.

    librdkafka has already stored offsets for every message it handed out, including a
    batch the caller never finished, so committing during teardown would skip exactly the
    alerts that still need reprocessing.
    """
    dataset = _build_dataset()
    consumer = _patch_consumer(monkeypatch, dataset, [])
    dataset._ensure_consumer()

    dataset.close()

    assert consumer.commits == []


def test_peek_sample_closes_consumer_on_interrupt(monkeypatch):
    """An interrupted peek must not leave a subscribed consumer behind.

    This is the notebook case: a KeyboardInterrupt out of peek_sample used to leave a live
    consumer holding the topic's partitions, so the next run joined the same group and was
    assigned nothing.
    """
    dataset = _build_dataset()
    consumer = _patch_consumer(monkeypatch, dataset, [], poll_error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        dataset.peek_sample()

    assert consumer.closed
    assert dataset._consumer is None


def test_peek_sample_closes_consumer_on_error(monkeypatch):
    """Any exception out of peek_sample releases the consumer."""
    dataset = _build_dataset()
    consumer = _patch_consumer(monkeypatch, dataset, [], poll_error=RuntimeError("broker exploded"))

    with pytest.raises(RuntimeError, match="broker exploded"):
        dataset.peek_sample()

    assert consumer.closed
    assert dataset._consumer is None


def test_peek_timeout_raises_with_diagnostic(monkeypatch):
    """peek_sample gives up instead of blocking forever, and says what it was waiting on."""
    h = hyrax.Hyrax()
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "test-topic"
    ds_config["peek_timeout"] = 0.05
    dataset = KafkaStreamDataset(h.config)
    consumer = _patch_consumer(monkeypatch, dataset, [])

    with pytest.raises(TimeoutError) as excinfo:
        dataset.peek_sample()

    message = str(excinfo.value)
    assert "peek_timeout" in message
    assert "test-topic" in message
    # An empty assignment is the signature of a consumer left over from an earlier run,
    # so the error has to point at the command that reveals it.
    assert "kafka-consumer-groups" in message
    assert consumer.closed


def test_peek_timeout_disabled_by_false(monkeypatch):
    """The TOML `false` sentinel restores the unbounded wait."""
    h = hyrax.Hyrax()
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "test-topic"
    ds_config["peek_timeout"] = False
    dataset = KafkaStreamDataset(h.config)

    assert dataset.peek_timeout is None


def test_offsets_committed_only_after_batch_is_consumed(monkeypatch):
    """Offsets advance when the caller comes back for more, not when a batch is handed out."""
    dataset = _build_dataset(batch_size=2, batch_flush_timeout=0.0)
    messages = [_make_message(f"id{i}", [[float(i)]]) for i in range(4)]
    consumer = _patch_consumer(monkeypatch, dataset, messages, stop_when_exhausted=True)

    iterator = iter(dataset)

    next(iterator)
    assert consumer.commits == []  # handed out, but the caller has not finished with it

    next(iterator)
    assert consumer.commits == [True]  # resuming committed the first batch, asynchronously


def test_abandoned_batch_is_not_committed(monkeypatch):
    """A batch the caller never finished stays uncommitted so it is redelivered.

    Closing the iterator stands in for the real failure: the caller raised, or broke out
    of the loop, leaving the generator suspended at its yield.
    """
    dataset = _build_dataset(batch_size=2, batch_flush_timeout=0.0)
    messages = [_make_message(f"id{i}", [[float(i)]]) for i in range(4)]
    consumer = _patch_consumer(monkeypatch, dataset, messages, stop_when_exhausted=True)

    iterator = iter(dataset)
    next(iterator)
    iterator.close()

    assert consumer.commits == []
    assert consumer.closed


def test_auto_commit_disabled():
    """The stream owns offset commits; librdkafka must not advance them on its own."""
    dataset = _build_dataset()

    assert dataset.consumer_config["enable.auto.commit"] is False
    # __iter__'s explicit commit relies on librdkafka having stored delivered offsets.
    assert dataset.consumer_config["enable.auto.offset.store"] is True


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


def test_extra_credentials(tmp_path):
    """Extra credentials are passed to the consumer constructor."""
    h = hyrax.Hyrax()
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "test-topic"
    extra_credentials = {
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "SCRAM-SHA-512",
        "sasl.username": "user",
        "sasl.password": "pass",
    }

    with open(tmp_path / "credentials.ini", "w") as f:
        for key, value in extra_credentials.items():
            f.write(f"'{key}' = '{value}'\n")
    ds_config["credentials_file"] = str(tmp_path / "credentials.ini")
    print(str(tmp_path / "credentials.ini"))

    dataset = KafkaStreamDataset(h.config)

    # The extra credentials should be present in the consumer config.
    for key, value in extra_credentials.items():
        assert dataset.consumer_config[key] == value


def test_consumer_options_reach_the_consumer_config():
    """Arbitrary librdkafka settings can be tuned without a credentials file."""
    h = hyrax.Hyrax()
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "test-topic"
    ds_config["consumer_options"] = {"max.poll.interval.ms": 600000, "session.timeout.ms": 45000}

    dataset = KafkaStreamDataset(h.config)

    assert dataset.consumer_config["max.poll.interval.ms"] == 600000
    assert dataset.consumer_config["session.timeout.ms"] == 45000


def test_consumer_options_win_over_credentials_file(tmp_path):
    """consumer_options is applied last, so it overrides the credentials file."""
    h = hyrax.Hyrax()
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "test-topic"

    credentials_path = tmp_path / "credentials.ini"
    credentials_path.write_text("'security.protocol' = 'SASL_PLAINTEXT'\n")
    ds_config["credentials_file"] = str(credentials_path)
    ds_config["consumer_options"] = {"security.protocol": "SASL_SSL"}

    dataset = KafkaStreamDataset(h.config)

    assert dataset.consumer_config["security.protocol"] == "SASL_SSL"
