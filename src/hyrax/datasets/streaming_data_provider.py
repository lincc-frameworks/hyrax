"""A ``DataProvider``-like wrapper that encapsulates a streaming dataset.

:class:`~hyrax.datasets.data_provider.DataProvider` is map-style — it assumes indexed
random access (``resolve_data(idx)``, joins, caching, augmentation, splits), none of
which a live stream supports. :class:`StreamingDataProvider` is its **iterable** sibling:
it wraps a single streaming dataset (e.g.
:class:`~hyrax.datasets.kafka_stream_dataset.KafkaStreamDataset`) and presents the same
surface the rest of Hyrax expects — ``collate`` (shared via
:class:`~hyrax.datasets.data_provider.CollationMixin`) and ``sample_data`` for model
pre-flighting — while delegating iteration to the wrapped stream.

The wrapped stream is a *dumb decoder* that yields ``list[dict]`` batches of flat sample
dicts. This provider owns the structuring: it extracts the object id (via
``primary_id_field``) and groups model inputs (via ``fields``) under the request
friendly-name, producing the per-sample shape ``collate`` expects::

    {"object_id": str, "<friendly_name>": {field: np.ndarray, ...}}

The single source of truth for ``primary_id_field`` and ``fields`` is ``[data_request]``.
"""

import logging

import numpy as np
import torch

from .data_provider import CollationMixin, DataProvider
from .dataset_registry import fetch_dataset_class

logger = logging.getLogger(__name__)


class StreamingDataProvider(CollationMixin, torch.utils.data.IterableDataset):
    """Wrap a single streaming ``IterableDataset`` behind a DataProvider-like surface.

    Parameters
    ----------
    config : dict
        The Hyrax runtime configuration.
    request : dict
        A single-entry data request group (``friendly_name -> definition``). The
        definition must specify ``dataset_class`` (a registered ``IterableDataset``) and
        ``primary_id_field``; ``fields`` is optional (derived from the first sample when
        omitted).
    """

    def __init__(self, config: dict, request: dict):
        self.config = config
        self.data_request = request

        if len(request) != 1:
            raise RuntimeError(
                "StreamingDataProvider supports exactly one streaming dataset per group; "
                f"got {len(request)} entries: {list(request)}."
            )

        friendly_name, definition = next(iter(request.items()))

        if definition.get("join_field"):
            raise RuntimeError(
                f"StreamingDataProvider does not support joined/secondary datasets "
                f"(request '{friendly_name}' sets 'join_field')."
            )

        dataset_class = definition.get("dataset_class")
        if not dataset_class:
            raise RuntimeError(
                f"Streaming data request '{friendly_name}' does not specify a 'dataset_class'."
            )

        dataset_cls = fetch_dataset_class(dataset_class)
        if not issubclass(dataset_cls, torch.utils.data.IterableDataset):
            raise RuntimeError(
                f"StreamingDataProvider requires an IterableDataset, but '{dataset_class}' is not one."
            )

        primary_id_field = definition.get("primary_id_field")
        if primary_id_field in (None, False):
            raise RuntimeError(f"Streaming data request '{friendly_name}' must set 'primary_id_field'.")

        self.friendly_name = friendly_name
        # Mirror the attribute names DataProvider exposes so shared/up-stream code that
        # references a "primary dataset" continues to work.
        self.primary_dataset = friendly_name
        self.primary_dataset_id_field_name = primary_id_field
        self.primary_id_field = primary_id_field

        # `fields` may be empty here; when so it is derived from the first sample.
        self.fields = list(definition.get("fields", []))

        # Instantiate the wrapped stream with any dataset-specific config overrides.
        dataset_specific_config = DataProvider._apply_configurations(config, definition)
        self._stream = dataset_cls(
            config=dataset_specific_config, data_location=definition.get("data_location")
        )

        self.prepped_datasets = {friendly_name: self._stream}

        # Collation wiring consumed by CollationMixin.collate.
        self.custom_collate_functions: dict = {}
        self.field_collate_functions: dict = {friendly_name: {}}
        stream_collate = getattr(self._stream, "collate", None)
        if callable(stream_collate):
            # A user subclass may define a dataset-level collate; honor it.
            self.custom_collate_functions[friendly_name] = stream_collate

        # If fields are known up front, register per-field collate hooks now; otherwise
        # this happens lazily once the first sample reveals the field names.
        if self.fields:
            self._register_field_collate_hooks()

    def _register_field_collate_hooks(self):
        """Detect ``collate_<field>`` methods on the wrapped stream for each field."""
        hooks = self.field_collate_functions[self.friendly_name]
        for field in self.fields:
            hook = getattr(self._stream, f"collate_{field}", None)
            hooks[field] = hook if callable(hook) else None

    def _structure(self, sample: dict) -> dict:
        """Turn a flat decoded sample into the per-sample shape ``collate`` expects."""
        if not self.fields:
            self.fields = [key for key in sample if key != self.primary_id_field]
            self._register_field_collate_hooks()
        data = {}
        for field in self.fields:
            arr = np.asarray(sample[field])
            data[field] = arr.astype(np.float32, copy=False) if np.issubdtype(arr.dtype, np.floating) else arr

        return {"object_id": str(sample[self.primary_id_field]), self.friendly_name: data}

    def __iter__(self):
        """Yield ``list[dict]`` batches of structured samples for ``collate_fn``."""
        for batch in self._stream:
            yield [self._structure(sample) for sample in batch]

    def sample_data(self) -> dict:
        """Return one structured sample for model pre-flighting (``setup_model``).

        Peeks a single message from the stream without losing it (it is replayed in the
        first batch) and structures it like any other sample.
        """
        return self._structure(self._stream.peek_sample())

    def stop(self):
        """Stop the underlying stream's iteration."""
        self._stream.stop()
