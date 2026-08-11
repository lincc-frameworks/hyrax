"""Streaming dataset that reads JSON messages from a Kafka topic.

:class:`KafkaStreamDataset` is a :class:`torch.utils.data.IterableDataset` intended
for live, open-ended inference (e.g. a telescope alert stream). Unlike the map-style
Hyrax datasets, it has no length: it polls a Kafka topic and yields batches as data
arrives.

The defining feature is **latency-bounded batching**. A PyTorch ``DataLoader`` cannot
emit a partial batch on a timeout, so the batching logic lives here in
:meth:`__iter__`: messages are accumulated and a batch is yielded as soon as *either*
``batch_size`` messages have arrived *or* ``batch_flush_timeout`` seconds have elapsed
since the first message of the current batch. This means inference still proceeds on a
short batch during quiet periods instead of blocking until the batch fills.

The stream is intentionally a **dumb decoder**: :meth:`_decode` returns the parsed JSON
object as a flat ``dict``. Extracting the object id and grouping model-input fields is
the responsibility of
:class:`~hyrax.datasets.streaming_data_provider.StreamingDataProvider`, which wraps this
dataset and knows the ``[data_request]`` (``primary_id_field`` and ``fields``). The
provider is what is passed to the DataLoader; configure it the normal way:

.. code-block:: python

    import hyrax

    hy = hyrax.Hyrax()
    hy.config["data_request"] = {
        "infer_stream": {
            "data": {
                "dataset_class": "KafkaStreamDataset",
                "primary_id_field": "object_id",
                "fields": ["image"],
            }
        }
    }
    hy.config["data_set"]["KafkaStreamDataset"]["topic"] = "ztf-alerts"

    with hy.infer_stream() as session:  # builds the provider + loader internally
        for batch, results in session:
            ...

Offsets are committed **after** a batch has been consumed downstream, never before; see
:meth:`KafkaStreamDataset.__iter__`. The consumer must therefore be closed deterministically
when iteration ends -- including when it ends because something raised -- which is what
:meth:`KafkaStreamDataset.close` is for. ``InferStreamSession`` calls it on every exit path,
so the ``with hy.infer_stream() as session`` form above is the supported way to drive the
stream.

Recovering / replaying a stream
-------------------------------
An unclosed consumer stays a member of its consumer group and keeps its partition
assignment, so the *next* run joins the same group, loses the rebalance, is assigned no
partitions and blocks forever. If you see "no partitions are assigned" in the logs, find
the lingering member with::

    kafka-consumer-groups --bootstrap-server localhost:9092 \\
        --group hyrax-infer-stream --describe

Separately, ``auto_offset_reset`` only applies to a group with *no* committed offset. Once
a group has consumed a topic, re-running resumes after the last commit rather than
replaying. To read a topic from the beginning again, either set a fresh ``group_id`` or
reset the existing group's offsets while no consumer is running::

    kafka-consumer-groups --bootstrap-server localhost:9092 \\
        --group hyrax-infer-stream --reset-offsets --to-earliest --all-topics --execute

.. warning::
    The stream uses a single in-process Kafka consumer, so the loader must run with
    ``num_workers = 0`` (the default applied by ``dist_data_loader``). With multiple
    workers each would open its own consumer and the :meth:`stop` signal would not
    propagate.
"""

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import toml
import torch

from .dataset_registry import HyraxDataset

logger = logging.getLogger(__name__)

# How often to log a "still waiting" warning while peek_sample() blocks on an empty topic.
PEEK_WARN_INTERVAL = 15.0


def _format_partitions(partitions) -> list[str]:
    """Render a list of ``TopicPartition`` as ``["topic[0]", "topic[1]"]`` for logging."""
    return [f"{part.topic}[{part.partition}]" for part in partitions]


class KafkaStreamDataset(HyraxDataset, torch.utils.data.IterableDataset):
    """Reads JSON messages from a Kafka topic and yields latency-bounded batches.

    Each Kafka message is expected to be a JSON object. :meth:`_decode` returns it as a
    flat ``dict`` (e.g. ``{"object_id": "...", "image": [...], ...}``); the wrapping
    :class:`~hyrax.datasets.streaming_data_provider.StreamingDataProvider` turns each
    flat sample into the structured form the collation + model machinery expect.
    """

    def __init__(self, config: dict, data_location=None):
        ds_config = config["data_set"]["KafkaStreamDataset"]

        # ``data_location``, when given, is a Kafka URI of the form
        # ``kafka://<host>:<port>[/<topic>]`` supplied inline by the data_request. It takes
        # precedence over the [data_set.KafkaStreamDataset] config; anything the URI omits
        # falls back to that config block.
        host_port = ""
        topic = ""
        if data_location:
            parsed = urlparse(data_location)
            if parsed.scheme == "kafka":
                host_port = parsed.netloc  # "broker.example.org:9092"
                topic = parsed.path.lstrip("/")  # "my-topic"

        self.bootstrap_servers = host_port or ds_config["bootstrap_servers"]
        self.topics = topic or ds_config["topics"]
        if not isinstance(self.topics, list) and isinstance(self.topics, str):
            self.topics = [self.topics]  # allow a single topic string for convenience

        # `topics` may still be the TOML `false` sentinel ("not set") here.
        if not self.topics:
            raise ValueError(
                "config['data_set']['KafkaStreamDataset']['topics'] must be set to a list of Kafka topics."
            )

        self.group_id = ds_config["group_id"]
        self.auto_offset_reset = ds_config["auto_offset_reset"]
        self.poll_timeout = float(ds_config["poll_timeout"])
        self.batch_flush_timeout = float(ds_config["batch_flush_timeout"])

        # Bound on how long peek_sample() waits for its first message. The TOML `false`
        # sentinel (or 0) means "wait indefinitely".
        peek_timeout = ds_config["peek_timeout"]
        self.peek_timeout = float(peek_timeout) if peek_timeout else None

        # The flush threshold is the configured DataLoader batch size.
        self.batch_size = config["data_loader"]["batch_size"]

        # Set from another thread (or session teardown) to end iteration; see stop().
        self._stop = threading.Event()

        # Single shared consumer, created lazily. Shared between peek_sample() and
        # __iter__ so a peeked message can be replayed into the first batch.
        self._consumer = None
        self._buffered: list[dict] = []

        credentials_file = ds_config.get("credentials_file")
        credentials_file_path = Path(credentials_file) if credentials_file else None

        self.consumer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": self.auto_offset_reset,
            # Offsets are committed by __iter__ once the caller has finished a batch.
            # Auto-commit would advance them as soon as consume() returns -- before
            # filtering, inference or writing -- so any failure downstream would silently
            # drop the in-flight alerts and they could never be replayed.
            "enable.auto.commit": False,
            # Left on deliberately, and __iter__ depends on it: librdkafka stores each
            # message's offset as it is delivered, so an explicit commit() after a batch
            # commits exactly the messages delivered so far and nothing beyond them. The
            # next consume() has not run yet at that point.
            "enable.auto.offset.store": True,
            # Identifies this process in `kafka-consumer-groups --describe`, which is how
            # you spot a consumer from an earlier run still holding the partitions.
            "client.id": f"hyrax-{socket.gethostname()}-{os.getpid()}",
        }

        if credentials_file_path and credentials_file_path.exists():
            credentials = toml.load(credentials_file_path)
            self.consumer_config.update(credentials)

        # Free-form librdkafka settings (max.poll.interval.ms, session.timeout.ms, ...).
        # Applied last so they win over both the defaults above and the credentials file.
        consumer_options = ds_config.get("consumer_options")
        if consumer_options:
            self.consumer_config.update(consumer_options)

        super().__init__(config, metadata_table=None)

    def stop(self):
        """Signal :meth:`__iter__` to flush any pending batch and stop iterating.

        This is only a signal. It does **not** release the Kafka connection -- use
        :meth:`close` for that.
        """
        self._stop.set()

    def close(self):
        """Stop iteration and close the Kafka consumer. Idempotent.

        Closing is what releases the partition assignment and removes this consumer from
        its group. Skipping it leaves a zombie group member:
        its librdkafka thread keeps running, it keeps the partitions, and the next run
        joining the same ``group_id`` can be assigned nothing at all and block forever.
        A suspended :meth:`__iter__` generator cannot clean itself up (nothing resumes it),
        and in a notebook the retained traceback keeps it from ever being collected, so
        the owner of the stream has to call this explicitly on error paths.

        ``_consumer is None`` is the guard, so a late second call -- the generator's
        ``finally`` running at collection time, long after the session closed -- is a
        no-op rather than a double close.

        Deliberately does **not** commit. Closing is reached on failure paths just as
        often as on clean ones, and librdkafka has already stored the offsets of every
        message it handed out -- including a batch the caller never finished. Committing
        here would advance past exactly the alerts that still need reprocessing. Since
        ``enable.auto.commit`` is off, ``Consumer.close()`` commits nothing on its own; it
        only flushes commits :meth:`__iter__` already issued for completed batches.
        """
        # Set before closing: __iter__'s while-guard must fail before it can call
        # consume() on a consumer that is already gone.
        self._stop.set()
        consumer, self._consumer = self._consumer, None
        if consumer is None:
            return

        try:
            consumer.close()
        except Exception as err:
            # Never let teardown replace the exception that triggered it.
            logger.warning(f"Error closing Kafka consumer: {err}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False

    def __len__(self):
        """A live stream has no length.

        Defined only so ``HyraxDataset.__init_subclass__`` (which requires a ``__len__``
        attribute) accepts the class. The iterable branch of ``dist_data_loader`` never
        calls it.
        """
        raise TypeError("KafkaStreamDataset is a live stream and has no length.")

    def _make_consumer(self):
        """Create and subscribe a Kafka consumer.

        Built lazily (not in ``__init__``) because Kafka consumers are not safe to fork
        or pickle across DataLoader workers.
        """
        try:
            from confluent_kafka import Consumer
        except ImportError as err:
            raise ImportError("KafkaStreamDataset requires the 'confluent-kafka' package. ") from err

        logger.info(
            f"Connecting Kafka consumer to {self.bootstrap_servers} "
            f"(group.id={self.group_id!r}, topics={self.topics})"
        )
        consumer = Consumer(self.consumer_config)

        # Logging-only callbacks: confluent_kafka performs the default (incremental)
        # assign/unassign itself when the callback does not do it.
        consumer.subscribe(self.topics, on_assign=self._on_assign, on_revoke=self._on_revoke)

        return consumer

    def _on_assign(self, consumer, partitions):
        """Log the partitions handed to this consumer on each rebalance."""
        if partitions:
            logger.info(f"Kafka partitions assigned: {_format_partitions(partitions)}")
        else:
            logger.warning(self._connection_diagnostic("Kafka assigned no partitions"))

    def _on_revoke(self, consumer, partitions):
        """Log the partitions taken from this consumer on each rebalance."""
        logger.info(f"Kafka partitions revoked: {_format_partitions(partitions)}")

    def _connection_diagnostic(self, prefix: str) -> str:
        """Describe what this consumer is connected to, and why it may see no messages."""
        assignment = []
        if self._consumer is not None:
            try:
                assignment = _format_partitions(self._consumer.assignment())
            except Exception:  # pragma: no cover - assignment() is a local lookup
                assignment = ["<unavailable>"]

        detail = (
            f"{prefix}: bootstrap.servers={self.bootstrap_servers!r}, "
            f"group.id={self.group_id!r}, topics={self.topics}, "
            f"assigned partitions={assignment or 'none'}"
        )
        if not assignment:
            detail += (
                ". With no partitions assigned no message can ever arrive. Another member "
                f"of group {self.group_id!r} is probably still holding them -- most often a "
                "previous run whose consumer was never closed. Check with "
                f"`kafka-consumer-groups --bootstrap-server {self.bootstrap_servers} "
                f"--group {self.group_id} --describe`."
            )
        return detail

    @staticmethod
    def _log_message_error(msg):
        """Log a message-level Kafka error instead of dropping it silently."""
        err = msg.error()
        try:
            from confluent_kafka import KafkaError

            if err.code() == KafkaError._PARTITION_EOF:
                logger.debug(f"Kafka reached end of partition: {err}")
                return
        except Exception:
            # A stand-in message, or an error object without a code(); fall through.
            pass
        logger.warning(f"Skipping Kafka message with error: {err}")

    def _ensure_consumer(self):
        """Return the shared consumer, creating it on first use."""
        if self._consumer is None:
            self._consumer = self._make_consumer()
        return self._consumer

    def _decode(self, msg) -> dict:
        """Decode a single Kafka message into a flat ``dict`` via JSON.

        The stream is intentionally a dumb decoder: it returns the parsed JSON object
        as-is. Extracting the object id and grouping model-input fields is the job of
        :class:`~hyrax.datasets.streaming_data_provider.StreamingDataProvider`.

        Parameters
        ----------
        msg : object
            A Kafka message whose ``value()`` is JSON bytes/str.

        Returns
        -------
        dict
            The parsed JSON object, e.g. ``{"object_id": "...", "image": [...], ...}``.
        """
        return json.loads(msg.value())

    def peek_sample(self) -> dict:
        """Return one decoded sample without removing it from the batch stream.

        Polls until a message arrives (or :meth:`stop` is set), decodes it, and buffers
        it so :meth:`__iter__` replays it as part of the first batch. Used to pre-flight
        the model architecture without losing a live message.

        Waiting is bounded by ``peek_timeout`` and reports what the consumer is connected
        to when it gives up. An unbounded silent wait here is indistinguishable from an
        unreachable broker, an empty topic, or a partition assignment lost to a consumer
        left over from an earlier run.

        Returns
        -------
        dict
            The flat decoded sample.

        Raises
        ------
        TimeoutError
            If ``peek_timeout`` elapses before a message arrives.
        RuntimeError
            If the stream is stopped before any message arrives.
        """
        consumer = self._ensure_consumer()
        started = time.monotonic()
        deadline = started + self.peek_timeout if self.peek_timeout else None
        next_warning = started + PEEK_WARN_INTERVAL

        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    raise TimeoutError(
                        self._connection_diagnostic(
                            f"No Kafka message arrived within peek_timeout ({self.peek_timeout}s)"
                        )
                    )
                if now >= next_warning:
                    next_warning = now + PEEK_WARN_INTERVAL
                    logger.warning(
                        self._connection_diagnostic(
                            f"Still waiting for a Kafka message after {now - started:.0f}s"
                        )
                    )

                msg = consumer.poll(self.poll_timeout)
                if msg is None:
                    continue
                if msg.error() is not None:
                    self._log_message_error(msg)
                    continue

                sample = self._decode(msg)
                self._buffered.append(sample)
                return sample

            raise RuntimeError("KafkaStreamDataset.peek_sample() stopped before a message arrived.")
        except BaseException:
            # There is no `finally` here on purpose: on success the consumer must stay
            # open for __iter__. On failure it must not -- and BaseException rather than
            # Exception because interrupting a hung notebook cell is the common case, and
            # a KeyboardInterrupt that leaked the consumer is exactly how a zombie group
            # member is created.
            self.close()
            raise

    def __iter__(self):
        """Poll Kafka and yield ``list[dict]`` batches with latency-bounded flushing.

        Offsets are committed only after the caller comes back for the next batch. Because
        this is a generator, that resumption *is* the signal that the previous batch was
        fully processed downstream -- inferred and written -- so it needs no cooperation
        from the caller. If the caller raises instead, the generator is never resumed, the
        commit never happens, and the batch is redelivered on the next run. That makes
        delivery at-least-once; duplicates after a crash are possible, dropped alerts are
        not.
        """
        if self._stop.is_set():
            return

        consumer = self._ensure_consumer()

        # Replay any peeked-but-not-yet-delivered messages ahead of the next batch.
        pending: list[dict] = list(self._buffered)
        self._buffered = []

        try:
            while not self._stop.is_set():
                # consume() blocks up to batch_flush_timeout and returns up to batch_size
                # messages: this is the latency-bounded batching, so inference still
                # proceeds on a short batch during quiet periods instead of blocking
                # until the batch fills.
                messages = consumer.consume(num_messages=self.batch_size, timeout=self.batch_flush_timeout)

                batch, pending = pending, []
                for msg in messages:
                    if msg.error() is None:
                        batch.append(self._decode(msg))
                    else:
                        self._log_message_error(msg)

                if batch:
                    yield batch
                    # Resumed, so the caller is done with that batch and it is finally
                    # safe to advance the committed offset. Asynchronous: a synchronous
                    # round trip per batch adds latency and eats into max.poll.interval.ms.
                    # Consumer.close() flushes whatever is still in flight.
                    self._commit(consumer)
        finally:
            # Routed through close() rather than consumer.close() so this is a no-op when
            # the owner already closed the stream and left this generator suspended.
            self.close()

    @staticmethod
    def _commit(consumer):
        """Commit the offsets of the batch the caller has just finished processing."""
        try:
            consumer.commit(asynchronous=True)
        except Exception as err:
            # Not fatal: the next batch's commit covers these offsets too. Worst case the
            # batch is redelivered, which is the intended failure direction.
            logger.warning(f"Failed to commit Kafka offsets: {err}")
