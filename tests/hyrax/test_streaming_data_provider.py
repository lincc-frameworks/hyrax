"""Tests for StreamingDataProvider — structuring, collation, sample_data, and routing.

Reuses the FakeConsumer pattern from ``test_kafka_stream_dataset.py``: the provider builds
a KafkaStreamDataset internally, so tests patch ``provider._stream._make_consumer``.

The provider reads every field through a ``get_<field>(sample)`` method on the wrapped
dataset rather than indexing the decoded sample dict, so the streams used here are
KafkaStreamDataset subclasses that define those getters.
"""

import logging

import numpy as np
import pytest
from test_kafka_stream_dataset import FakeConsumer, _make_message
from torch.utils.data import DataLoader

import hyrax
from hyrax.datasets.data_provider import DataProvider
from hyrax.datasets.kafka_stream_dataset import KafkaStreamDataset
from hyrax.datasets.streaming_data_provider import StreamingDataProvider
from hyrax.pytorch_ignite import dist_data_loader, setup_dataset


class _GetterStream(KafkaStreamDataset):
    """KafkaStreamDataset subclass exposing ``get_<field>`` accessors over flat samples.

    ``get_flux`` is deliberately *derived* — the decoded messages carry no ``flux`` key —
    so tests can tell getter-mediated access apart from a plain dict lookup.
    """

    def get_image(self, sample):
        """Return the image payload of one decoded sample."""
        return sample["image"]

    def get_flux(self, sample):
        """Compute a scalar flux from the image; there is no `flux` key in the message."""
        return float(np.sum(sample["image"]))

    def get_object_id(self, sample):
        """Return the object id as a string, regardless of its original type."""
        return str(sample["object_id"])


class _HookedStream(_GetterStream):
    """Getter stream that also defines a per-field collate hook for `image`."""

    def collate_image(self, samples):
        """Stack images and also emit a boolean mask of the same shape."""
        arr = np.stack([s["image"] for s in samples], axis=0)
        return {"image": arr, "image_mask": np.ones_like(arr, dtype=bool)}


def _build_provider(
    batch_size=5, flush=100.0, fields=("image",), primary_id="object_id", dataset_class="_GetterStream"
):
    """Construct a StreamingDataProvider wrapping a KafkaStreamDataset."""
    h = hyrax.Hyrax()
    h.config["data_loader"]["batch_size"] = batch_size
    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "test-topic"
    ds_config["batch_flush_timeout"] = flush
    request = {
        "data": {
            "dataset_class": dataset_class,
            "primary_id_field": primary_id,
            "fields": list(fields),
        }
    }
    return StreamingDataProvider(h.config, request)


def _patch_stream(monkeypatch, provider, messages, stop_when_exhausted=True):
    """Inject a FakeConsumer into the provider's wrapped stream."""
    stream = provider._stream
    on_exhausted = stream.stop if stop_when_exhausted else None
    consumer = FakeConsumer(messages, on_exhausted=on_exhausted)
    monkeypatch.setattr(stream, "_make_consumer", lambda: consumer)
    return consumer


def test_structure_shapes_a_flat_sample():
    """_structure pulls object_id out and groups fields under the friendly-name."""
    provider = _build_provider(fields=("image",))
    structured = provider._structure({"object_id": "abc", "image": [[1.0, 2.0]]})

    assert set(structured.keys()) == {"object_id", "data"}
    assert structured["object_id"] == "abc"
    assert structured["data"]["image"].shape == (1, 2)
    assert structured["data"]["image"].dtype == np.float32


def test_structure_reads_fields_through_dataset_getters():
    """Field values come from `get_<field>(sample)`, not from indexing the sample dict."""
    provider = _build_provider(fields=("image", "flux"))
    # The decoded message has no `flux` key — only `_GetterStream.get_flux` can produce it.
    structured = provider._structure({"object_id": "abc", "image": [[1.0, 2.0], [3.0, 4.0]]})

    assert structured["data"]["flux"] == np.float32(10.0)
    assert structured["data"]["image"].shape == (2, 2)


def test_getters_are_cached_from_the_stream_instance():
    """Every `get_*` method on the stream is cached as a bound method, regardless of `fields`."""
    provider = _build_provider(fields=("image",))

    # All getters are cached, even though only `image` was requested...
    assert set(provider.dataset_getters["data"]) == {"image", "flux", "object_id"}
    # ...and each one is bound to the wrapped stream instance.
    for getter in provider.dataset_getters["data"].values():
        assert getter.__self__ is provider._stream

    # The requested `fields` still control what `_structure` emits.
    assert provider.fields == ["image"]
    assert set(provider._structure({"object_id": "a", "image": [[1.0]]})["data"]) == {"image"}


def test_fields_derived_from_getters_when_not_configured():
    """With no `fields` in the request, they are derived from the stream's `get_*` methods."""
    provider = _build_provider(fields=())

    # Derived at construction time — no sample needed. `dir()` orders them alphabetically.
    assert provider.fields == ["flux", "image", "object_id"]

    structured = provider._structure({"object_id": "x", "image": [[2.0]]})
    assert set(structured["data"]) == {"flux", "image", "object_id"}


def test_requested_field_without_a_getter_raises():
    """A requested field with no matching `get_<field>` method fails when structuring."""
    provider = _build_provider(fields=("image", "no_such_field"))

    with pytest.raises(KeyError, match="no_such_field"):
        provider._structure({"object_id": "a", "image": [[1.0]]})


def test_error_logged_when_stream_has_no_getters(caplog):
    """A stream with no `get_*` methods is a dataset-definition error and is logged as one."""
    with caplog.at_level(logging.ERROR, logger="hyrax.datasets.streaming_data_provider"):
        provider = _build_provider(fields=(), dataset_class="KafkaStreamDataset")

    assert provider.dataset_getters["data"] == {}
    assert provider.fields == []
    assert "No `get_*` methods were found" in caplog.text
    assert "KafkaStreamDataset" in caplog.text


def test_collate_matches_infer_contract():
    """collate (from CollationMixin) yields object_id ndarray + grouped data fields."""
    provider = _build_provider(fields=("image",))
    batch = [provider._structure({"object_id": f"id{i}", "image": [[float(i), 0.0]]}) for i in range(4)]

    collated = provider.collate(batch)

    assert set(collated.keys()) == {"object_id", "data"}
    assert list(collated["object_id"]) == ["id0", "id1", "id2", "id3"]
    assert collated["object_id"].dtype.kind in ("U", "S")
    assert collated["data"]["image"].shape == (4, 1, 2)


def test_sample_data_returns_structured_and_is_not_lost(monkeypatch):
    """sample_data peeks one structured sample that still appears in the first batch."""
    provider = _build_provider(batch_size=5, flush=0.0, fields=("image",))
    messages = [_make_message("first", [[1.0]]), _make_message("second", [[2.0]])]
    _patch_stream(monkeypatch, provider, messages)

    sample = provider.sample_data()
    assert sample["object_id"] == "first"
    assert sample["data"]["image"].shape == (1, 1)

    batches = list(provider)
    assert len(batches) == 1
    assert [s["object_id"] for s in batches[0]] == ["first", "second"]


def test_dist_data_loader_end_to_end(monkeypatch):
    """The provider flows through dist_data_loader and yields collated batch dicts."""
    provider = _build_provider(batch_size=2, fields=("image", "flux"))
    messages = [_make_message(f"id{i}", [[float(i)]]) for i in range(2)]
    _patch_stream(monkeypatch, provider, messages)

    loader = dist_data_loader(provider, provider.config)
    assert isinstance(loader, DataLoader)
    assert loader.batch_size is None
    assert loader.collate_fn == provider.collate

    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]
    assert list(batch["object_id"]) == ["id0", "id1"]
    assert batch["data"]["image"].shape == (2, 1, 1)
    # The derived (getter-only) field survives the whole path to the batch dict.
    assert list(batch["data"]["flux"]) == [0.0, 1.0]


def test_requires_single_dataset():
    """A streaming group must contain exactly one dataset (no joins)."""
    h = hyrax.Hyrax()
    h.config["data_set"]["KafkaStreamDataset"]["topics"] = "t"
    request = {
        "a": {"dataset_class": "_GetterStream", "primary_id_field": "object_id"},
        "b": {"dataset_class": "_GetterStream", "primary_id_field": "object_id"},
    }
    with pytest.raises(RuntimeError, match="exactly one"):
        StreamingDataProvider(h.config, request)


def test_requires_iterable_dataset():
    """A map-style dataset cannot be wrapped by StreamingDataProvider."""
    h = hyrax.Hyrax()
    request = {"data": {"dataset_class": "HyraxRandomDataset", "primary_id_field": "object_id"}}
    with pytest.raises(RuntimeError, match="IterableDataset"):
        StreamingDataProvider(h.config, request)


def test_requires_primary_id_field():
    """The request must declare which field is the object id."""
    h = hyrax.Hyrax()
    h.config["data_set"]["KafkaStreamDataset"]["topics"] = "t"
    request = {"data": {"dataset_class": "_GetterStream"}}
    with pytest.raises(RuntimeError, match="primary_id_field"):
        StreamingDataProvider(h.config, request)


def test_field_collate_hook_is_honored():
    """A collate_<field> method on the wrapped stream is used during collation."""
    provider = _build_provider(fields=("image",), dataset_class="_HookedStream")
    batch = [provider._structure({"object_id": f"id{i}", "image": [[float(i)]]}) for i in range(3)]

    collated = provider.collate(batch)

    assert collated["data"]["image"].shape == (3, 1, 1)
    assert "image_mask" in collated["data"]
    assert collated["data"]["image_mask"].dtype == bool


def test_field_collate_hooks_registered_for_derived_fields():
    """Hooks are registered for getter-derived fields too, not just configured ones."""
    provider = _build_provider(fields=(), dataset_class="_HookedStream")
    hooks = provider.field_collate_functions["data"]

    assert set(hooks) == {"flux", "image", "object_id"}
    assert hooks["image"] == provider._stream.collate_image
    assert hooks["flux"] is None

    collated = provider.collate([provider._structure({"object_id": "a", "image": [[3.0]]})])
    assert "image_mask" in collated["data"]
    assert collated["data"]["flux"] == np.float32(3.0)


def test_setup_dataset_routes_streaming_vs_map():
    """setup_dataset picks StreamingDataProvider for iterable datasets, DataProvider otherwise."""
    h = hyrax.Hyrax()
    h.config["data_set"]["KafkaStreamDataset"]["topics"] = "t"
    h.config["data_request"] = {
        "stream": {
            "data": {
                "dataset_class": "_GetterStream",
                "primary_id_field": "object_id",
                "fields": ["image"],
            }
        },
        "static": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "primary_id_field": "object_id",
                "fields": ["image"],
            }
        },
    }

    providers = setup_dataset(h.config)

    assert isinstance(providers["stream"], StreamingDataProvider)
    assert isinstance(providers["static"], DataProvider)
    assert not isinstance(providers["static"], StreamingDataProvider)
