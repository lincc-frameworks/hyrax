"""Tests for StreamingDataProvider — structuring, collation, sample_data, and routing.

Reuses the FakeConsumer pattern from ``test_kafka_stream_dataset.py``: the provider builds
a KafkaStreamDataset internally, so tests patch ``provider._stream._make_consumer``.
"""

import numpy as np
import pytest
from test_kafka_stream_dataset import FakeConsumer, _make_message
from torch.utils.data import DataLoader

import hyrax
from hyrax.datasets.data_provider import DataProvider
from hyrax.datasets.kafka_stream_dataset import KafkaStreamDataset
from hyrax.datasets.streaming_data_provider import StreamingDataProvider
from hyrax.pytorch_ignite import dist_data_loader, setup_dataset


def _build_provider(
    batch_size=5, flush=100.0, fields=("image",), primary_id="object_id", dataset_class="KafkaStreamDataset"
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


def test_collate_matches_infer_contract():
    """collate (from CollationMixin) yields object_id ndarray + grouped data fields."""
    provider = _build_provider(fields=("image",))
    batch = [provider._structure({"object_id": f"id{i}", "image": [[float(i), 0.0]]}) for i in range(4)]

    collated = provider.collate(batch)

    assert set(collated.keys()) == {"object_id", "data"}
    assert list(collated["object_id"]) == ["id0", "id1", "id2", "id3"]
    assert collated["object_id"].dtype.kind in ("U", "S")
    assert collated["data"]["image"].shape == (4, 1, 2)


def test_fields_derived_when_not_configured(monkeypatch):
    """With no `fields` in the request, they are derived from the first sample."""
    provider = _build_provider(fields=())
    provider._structure({"object_id": "x", "image": [[1.0]], "flux": 3.0})

    assert provider.fields == ["image", "flux"]


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
    provider = _build_provider(batch_size=2, fields=("image",))
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


def test_requires_single_dataset():
    """A streaming group must contain exactly one dataset (no joins)."""
    h = hyrax.Hyrax()
    h.config["data_set"]["KafkaStreamDataset"]["topics"] = "t"
    request = {
        "a": {"dataset_class": "KafkaStreamDataset", "primary_id_field": "object_id"},
        "b": {"dataset_class": "KafkaStreamDataset", "primary_id_field": "object_id"},
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
    request = {"data": {"dataset_class": "KafkaStreamDataset"}}
    with pytest.raises(RuntimeError, match="primary_id_field"):
        StreamingDataProvider(h.config, request)


class _HookedStream(KafkaStreamDataset):
    """KafkaStreamDataset subclass with a per-field collate hook for `image`."""

    def collate_image(self, samples):
        """Stack images and also emit a boolean mask of the same shape."""
        arr = np.stack([s["image"] for s in samples], axis=0)
        return {"image": arr, "image_mask": np.ones_like(arr, dtype=bool)}


def test_field_collate_hook_is_honored():
    """A collate_<field> method on the wrapped stream is used during collation."""
    provider = _build_provider(fields=("image",), dataset_class="_HookedStream")
    batch = [provider._structure({"object_id": f"id{i}", "image": [[float(i)]]}) for i in range(3)]

    collated = provider.collate(batch)

    assert collated["data"]["image"].shape == (3, 1, 1)
    assert "image_mask" in collated["data"]
    assert collated["data"]["image_mask"].dtype == bool


def test_setup_dataset_routes_streaming_vs_map():
    """setup_dataset picks StreamingDataProvider for iterable datasets, DataProvider otherwise."""
    h = hyrax.Hyrax()
    h.config["data_set"]["KafkaStreamDataset"]["topics"] = "t"
    h.config["data_request"] = {
        "stream": {
            "data": {
                "dataset_class": "KafkaStreamDataset",
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
