import sys
from types import SimpleNamespace

import pytest

from hyrax.datasets.mmu_dataset import MultimodalUniverseDataset


class _FakeMapDataset:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, idx):
        return self._rows[idx]

    def __len__(self):
        return len(self._rows)

    def with_format(self, fmt):
        """Mimic HuggingFace Dataset.with_format; just return self for tests."""
        return self


class _FakeIterableDataset:
    """Fake streaming (iterable) dataset that supports with_format."""

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def with_format(self, fmt):
        """Mimic HuggingFace IterableDataset.with_format; just return self for tests."""
        return self


def _install_fake_datasets_module(monkeypatch, fake_load_dataset):
    fake_module = SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)


def test_mmu_dataset_uses_hf_uri_and_max_samples(monkeypatch):
    """Test that hf:// URIs are stripped and max_samples limits the dataset size."""
    calls = []
    rows = [
        {"object_id": "a1", "image": [1, 2], "label-name": "galaxy"},
        {"object_id": "a2", "image": [3, 4], "label-name": "star"},
    ]

    def fake_load_dataset(path, split, streaming):
        calls.append({"path": path, "split": split, "streaming": streaming})
        return _FakeMapDataset(rows)

    _install_fake_datasets_module(monkeypatch, fake_load_dataset)

    dataset = MultimodalUniverseDataset(
        config={"data_set": {"MultimodalUniverseDataset": {"split": "train", "max_samples": 2}}},
        data_location="hf://MultimodalUniverse/galaxy10_decals",
    )

    assert len(dataset) == 2
    assert calls[0]["path"] == "MultimodalUniverse/galaxy10_decals"
    assert calls[0]["split"] == "train"
    assert calls[0]["streaming"] is False
    assert dataset.get_label_name(0) == "galaxy"
    assert getattr(dataset, "get_label-name")(0) == "galaxy"


def test_mmu_dataset_enforces_hard_limit_when_split_is_pre_sliced(monkeypatch):
    """Test that max_samples enforces an additional hard limit even when split is pre-sliced."""
    calls = []
    rows = [
        {"object_id": "a1", "value": 1},
        {"object_id": "a2", "value": 2},
    ]

    def fake_load_dataset(path, split, streaming):
        calls.append({"path": path, "split": split, "streaming": streaming})
        return _FakeMapDataset(rows)

    _install_fake_datasets_module(monkeypatch, fake_load_dataset)

    dataset = MultimodalUniverseDataset(
        config={"data_set": {"MultimodalUniverseDataset": {"split": "train[:100]", "max_samples": 1}}},
        data_location="hf://MultimodalUniverse/plasticc",
    )

    assert calls[0]["split"] == "train[:100]"
    assert len(dataset) == 1
    assert dataset.get_value(0) == 1


def test_mmu_dataset_streaming_requires_max_samples(monkeypatch):
    """Test that streaming=True without max_samples raises a ValueError."""

    def fake_load_dataset(path, split, streaming):
        return _FakeIterableDataset([{"object_id": "1", "flux": [0.1, 0.2]}])

    _install_fake_datasets_module(monkeypatch, fake_load_dataset)

    with pytest.raises(ValueError):
        MultimodalUniverseDataset(
            config={"data_set": {"MultimodalUniverseDataset": {"split": "train", "streaming": True}}},
            data_location="hf://MultimodalUniverse/plasticc",
        )
