from pathlib import Path

import pandas as pd
import pytest
from nested_pandas import NestedFrame

from hyrax.datasets.nested_pandas_dataset import NestedPandasDataset


@pytest.fixture
def nested_pandas_parquet(tmp_path):
    """Create a tiny nested-pandas parquet file on disk."""
    source = pd.DataFrame(
        {
            "object_id": ["obj_1", "obj_2", "obj_3"],
            "ra": [10.5, 20.25, 30.75],
            "dec": [-1.0, -2.0, -3.0],
            "time": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
            "flux": [[100.0, 101.5], [200.25], [300.0, 301.0, 302.0]],
        }
    )
    frame = NestedFrame.from_flat(
        source,
        base_columns=["ra", "dec"],
        nested_columns=["time", "flux"],
        on="object_id",
        name="lightcurve",
    )
    data_path = tmp_path / "nested.parquet"
    frame.to_parquet(data_path)
    return data_path


def test_nested_pandas_dataset_loads_parquet_and_exposes_getters(nested_pandas_parquet):
    """Verify a real nested parquet file loads and exposes the expected getters."""
    dataset = NestedPandasDataset(
        config={
            "data_set": {
                "NestedPandasDataset": {
                    "read_parquet_kwargs": {},
                }
            }
        },
        data_location=nested_pandas_parquet,
    )

    assert len(dataset) == 3
    assert dataset.get_object_id(0) == "obj_1"
    assert dataset.get_ra(1) == pytest.approx(20.25)
    assert dataset.get_dec(2) == pytest.approx(-3.0)
    assert dataset.get_lightcurve_time(0) == [1.0, 2.0]
    assert dataset.get_lightcurve_flux(2) == [300.0, 301.0, 302.0]
    assert dataset.get_lightcurve(1) == {"time": [3.0], "flux": [200.25]}


def test_nested_pandas_dataset_passes_through_read_parquet_kwargs(monkeypatch):
    """Verify ``read_parquet_kwargs`` are forwarded to ``nested_pandas.read_parquet``."""
    captured = {}
    source = pd.DataFrame(
        {
            "object_id": ["obj_1"],
            "ra": [10.5],
            "dec": [-1.0],
            "time": [[1.0, 2.0]],
            "flux": [[100.0, 101.5]],
        }
    )
    frame = NestedFrame.from_flat(
        source,
        base_columns=["ra", "dec"],
        nested_columns=["time", "flux"],
        on="object_id",
        name="lightcurve",
    )

    def fake_read_parquet(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return frame

    monkeypatch.setattr("nested_pandas.read_parquet", fake_read_parquet)

    dataset = NestedPandasDataset(
        config={
            "data_set": {
                "NestedPandasDataset": {
                    "read_parquet_kwargs": {"columns": ["object_id", "ra"], "autocast_list": False},
                }
            }
        },
        data_location=Path("/tmp/nested.parquet"),
    )

    assert captured["data"] == "/tmp/nested.parquet"
    assert captured["kwargs"] == {"columns": ["object_id", "ra"], "autocast_list": False}
    assert len(dataset) == 1


def test_nested_pandas_dataset_requires_data_location():
    """Verify the dataset fails fast when no parquet location is provided."""
    with pytest.raises(ValueError):
        NestedPandasDataset(config={"data_set": {"NestedPandasDataset": {"read_parquet_kwargs": {}}}})
