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

.. warning::
    The stream uses a single in-process Kafka consumer, so the loader must run with
    ``num_workers = 0`` (the default applied by ``dist_data_loader``). With multiple
    workers each would open its own consumer and the :meth:`stop` signal would not
    propagate.
"""

import json
import logging
from pathlib import Path
import threading
from urllib.parse import urlparse

import toml
import torch

from .dataset_registry import HyraxDataset

logger = logging.getLogger(__name__)


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
        # ``kafka://<host>:<port>/<topic>`` supplied inline by the data_request. It takes
        # precedence over the [data_set.KafkaStreamDataset] config; anything the URI omits
        # falls back to that config block.
        host_port = ""
        topic = ""
        if data_location:
            parsed = urlparse(data_location)
            host_port = parsed.netloc  # "broker.example.org:9092"
            topic = parsed.path.lstrip("/")  # "lsst_topic"

        self.bootstrap_servers = host_port or ds_config["bootstrap_servers"]
        topic = topic or ds_config["topic"]

        # `topic` may still be the TOML `false` sentinel ("not set") here.
        if not topic:
            raise ValueError(
                "config['data_set']['KafkaStreamDataset']['topic'] must be set to the Kafka topic to consume."
            )

        self.topic = topic
        self.group_id = ds_config["group_id"]
        self.auto_offset_reset = ds_config["auto_offset_reset"]
        self.poll_timeout = float(ds_config["poll_timeout"])
        self.batch_flush_timeout = float(ds_config["batch_flush_timeout"])

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
        }

        if credentials_file_path and credentials_file_path.exists():
            credentials = toml.load(credentials_file_path)
            self.consumer_config.update(credentials)

        super().__init__(config, metadata_table=None)

    def stop(self):
        """Signal :meth:`__iter__` to flush any pending batch and stop iterating."""
        self._stop.set()

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
            raise ImportError(
                "KafkaStreamDataset requires the 'confluent-kafka' package. "
                "Install it with the streaming extra: pip install 'hyrax[stream]'."
            ) from err

        consumer = Consumer(self.consumer_config)

        consumer.subscribe([self.topic])

        return consumer

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

        Returns
        -------
        dict
            The flat decoded sample.

        Raises
        ------
        RuntimeError
            If the stream is stopped before any message arrives.
        """
        consumer = self._ensure_consumer()
        while not self._stop.is_set():
            msg = consumer.poll(self.poll_timeout)
            if msg is not None and msg.error() is None:
                sample = self._decode(msg)
                self._buffered.append(sample)
                return sample
        raise RuntimeError("KafkaStreamDataset.peek_sample() stopped before a message arrived.")

    def __iter__(self):
        """Poll Kafka and yield ``list[dict]`` batches with latency-bounded flushing."""
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

                if batch:
                    yield batch
        finally:
            consumer.close()
            self._consumer = None
