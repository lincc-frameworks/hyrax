import sys
import types

import pandas as pd
import pytest

from hyrax.datasets.hats_dataset import HATSPartitionIndex, HyraxHATSDataset


class _FakePartition:
    def __init__(self, df):
        self._df = df

    def __getitem__(self, columns):
        return _FakePartition(self._df[columns])

    def compute(self):
        return self._df


class _FakeCountResult:
    def __init__(self, counts):
        self._counts = counts

    def compute(self):
        return self._counts


class _FakeDDF:
    def __init__(self, partitions):
        self._partitions = partitions
        self.columns = partitions[0].columns if partitions else []

    def map_partitions(self, fn):
        return _FakeCountResult([len(partition) for partition in self._partitions])

    def get_partition(self, partition_id):
        return _FakePartition(self._partitions[partition_id])


class _FakeCatalog:
    def __init__(self, partitions):
        self._ddf = _FakeDDF(partitions)
        self.columns = partitions[0].columns if partitions else []
        self.compute_calls = 0

    def compute(self):
        self.compute_calls += 1
        return pd.concat(self._ddf._partitions, ignore_index=True)


@pytest.fixture
def fake_catalog():
    p0 = pd.DataFrame({"object_id": [1, 2], "coord_ra": [11.0, 12.0], "mag-r": [20.1, 20.2]})
    p1 = pd.DataFrame({"object_id": [3, 4, 5], "coord_ra": [13.0, 14.0, 15.0], "mag-r": [20.3, 20.4, 20.5]})
    return _FakeCatalog([p0, p1])


def _config_with_hats_options():
    return {
        "data_request": {
            "train": {
                "data": {
                    "dataset_class": "HyraxHATSDataset",
                    "data_location": "/fake/path",
                    "fields": ["coord_ra"],
                    "primary_id_field": "object_id",
                    "hats": {
                        "bundle_size": 2,
                        "max_cached_bundles": 2,
                        "project_columns": "auto",
                        "strict_metadata": True,
                    },
                }
            }
        }
    }


def test_partition_index_resolve(fake_catalog):
    index = HATSPartitionIndex(fake_catalog)

    assert index.total_rows == 5
    assert index.resolve(0) == (0, 0)
    assert index.resolve(1) == (0, 1)
    assert index.resolve(2) == (1, 0)
    assert index.resolve(4) == (1, 2)

    with pytest.raises(IndexError):
        index.resolve(5)


def test_hats_dataset_prefers_open_catalog(monkeypatch, fake_catalog):
    fake_lsdb = types.SimpleNamespace(
        open_catalog=lambda _loc: fake_catalog,
        read_hats=lambda _loc: pytest.fail("read_hats should not be called when open_catalog succeeds"),
    )
    monkeypatch.setitem(sys.modules, "lsdb", fake_lsdb)

    dataset = HyraxHATSDataset(_config_with_hats_options(), data_location="/fake/path")

    assert len(dataset) == 5
    assert dataset.get_object_id(3) == 4
    assert getattr(dataset, "get_mag-r")(4) == pytest.approx(20.5)
    assert fake_catalog.compute_calls == 0


def test_hats_dataset_falls_back_to_read_hats(monkeypatch, fake_catalog):
    def _boom(_loc):
        raise RuntimeError("open failed")

    fake_lsdb = types.SimpleNamespace(open_catalog=_boom, read_hats=lambda _loc: fake_catalog)
    monkeypatch.setitem(sys.modules, "lsdb", fake_lsdb)

    dataset = HyraxHATSDataset(_config_with_hats_options(), data_location="/fake/path")
    assert len(dataset) == 5
    assert dataset.get_coord_ra(2) == pytest.approx(13.0)


def test_hats_dataset_uses_bundle_cache(monkeypatch, fake_catalog):
    fake_lsdb = types.SimpleNamespace(open_catalog=lambda _loc: fake_catalog, read_hats=lambda _loc: fake_catalog)
    monkeypatch.setitem(sys.modules, "lsdb", fake_lsdb)

    dataset = HyraxHATSDataset(_config_with_hats_options(), data_location="/fake/path")

    _ = dataset.get_coord_ra(0)
    misses_after_first = dataset._accessor.cache.misses
    _ = dataset.get_object_id(1)
    hits_after_second = dataset._accessor.cache.hits

    assert misses_after_first == 1
    assert hits_after_second >= 1
