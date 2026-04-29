from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hyrax

nested_pandas = pytest.importorskip("nested_pandas")


@pytest.fixture(scope="function")
def nested_parquet_path(tmp_path: Path) -> Path:
    """Write a small nested-pandas NestedFrame to a parquet file and return its path."""
    base = pd.DataFrame(
        {
            "object_id": [1001, 1002, 1003],
            "ra": [150.1, 150.2, 150.3],
        },
        index=[0, 1, 2],
    )

    lightcurve = pd.DataFrame(
        {
            "mjd": [59000.0, 59001.0, 59002.0, 59003.0, 59004.0, 59005.0],
            "flux": [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        },
        index=[0, 0, 1, 1, 2, 2],
    )

    nf = nested_pandas.NestedFrame(base).join_nested(lightcurve, "lightcurve")

    out_path = tmp_path / "sample.parquet"
    nf.to_parquet(out_path)
    return out_path


def _make_dataset(parquet_path: Path):
    from hyrax.datasets.nested_pandas_dataset import NestedPandasDataset

    config = {
        "data_set": {
            "NestedPandasDataset": {
                "read_parquet_kwargs": {},
            }
        }
    }
    return NestedPandasDataset(config=config, data_location=parquet_path)


def test_length(nested_parquet_path):
    """Dataset reports the number of base rows."""
    dataset = _make_dataset(nested_parquet_path)
    assert len(dataset) == 3


def test_base_column_getters(nested_parquet_path):
    """Base (scalar) columns are exposed via ``get_<column>``."""
    dataset = _make_dataset(nested_parquet_path)
    assert dataset.get_object_id(0) == 1001
    assert dataset.get_ra(2) == pytest.approx(150.3)


def test_nested_column_getter_returns_dict_of_arrays(nested_parquet_path):
    """A nested column's getter returns one dict of arrays per object."""
    dataset = _make_dataset(nested_parquet_path)
    lc = dataset.get_lightcurve(1)
    assert isinstance(lc, dict)
    assert set(lc.keys()) == {"mjd", "flux"}
    assert isinstance(lc["mjd"], np.ndarray)
    np.testing.assert_allclose(lc["mjd"], [59002.0, 59003.0])
    np.testing.assert_allclose(lc["flux"], [20.0, 21.0])


def test_missing_data_location_raises():
    """Constructing without ``data_location`` raises ``ValueError``."""
    from hyrax.datasets.nested_pandas_dataset import NestedPandasDataset

    config = {"data_set": {"NestedPandasDataset": {"read_parquet_kwargs": {}}}}
    with pytest.raises(ValueError):
        NestedPandasDataset(config=config, data_location=None)


def test_through_hyrax_data_request(nested_parquet_path):
    """The dataset is reachable through a Hyrax data_request."""
    h = hyrax.Hyrax()
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "NestedPandasDataset",
                "data_location": str(nested_parquet_path),
                "fields": ["ra", "lightcurve"],
                "primary_id_field": "object_id",
                "split_fraction": 1.0,
            }
        }
    }

    dataset = h.prepare()
    assert len(dataset["train"]) == 3
