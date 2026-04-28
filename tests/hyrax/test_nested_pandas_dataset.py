from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from nested_pandas.nestedframe import NestedFrame

from hyrax.datasets.nested_pandas_dataset import NestedPandasDataset


@pytest.fixture
def nested_parquet_file(tmp_path):
    """Create a minimal nested-pandas parquet file on disk."""
    flat_data = pd.DataFrame(
        {
            "object_id": ["obj_1", "obj_1", "obj_2", "obj_2", "obj_2"],
            "ra": [10.0, 10.0, 20.0, 20.0, 20.0],
            "dec": [-1.0, -1.0, 0.5, 0.5, 0.5],
            "time": [1.0, 2.0, 1.5, 2.5, 3.5],
            "flux": [100.0, 101.0, 50.0, 51.0, 52.0],
            "band": ["g", "r", "g", "r", "i"],
        }
    )
    nested_frame = NestedFrame.from_flat(
        flat_data,
        base_columns=["object_id", "ra", "dec"],
        nested_columns=["time", "flux", "band"],
        on="object_id",
        name="lightcurve",
    )
    parquet_path = tmp_path / "sample_nested_frame.parquet"
    nested_frame.to_parquet(parquet_path)
    return parquet_path


def test_nested_pandas_dataset_reads_parquet_and_registers_nested_getters(nested_parquet_file):
    """Verify scalar, nested-table, and nested-subcolumn getters."""
    dataset = NestedPandasDataset(
        config={"data_set": {"NestedPandasDataset": {"read_parquet_kwargs": {}}}},
        data_location=nested_parquet_file,
    )

    assert len(dataset) == 2
    assert dataset.get_object_id(0) == "obj_1"
    assert dataset.get_ra(1) == 20.0

    nested_table = dataset.get_lightcurve(0)
    assert list(nested_table.columns) == ["time", "flux", "band"]
    np.testing.assert_allclose(dataset.get_lightcurve__time(0), np.array([1.0, 2.0]))
    np.testing.assert_allclose(dataset.get_lightcurve__flux(1), np.array([50.0, 51.0, 52.0]))
    np.testing.assert_array_equal(dataset.get_lightcurve__band(1), np.array(["g", "r", "i"]))


def test_nested_pandas_dataset_passes_read_parquet_kwargs(nested_parquet_file):
    """Verify config read kwargs are forwarded to nested_pandas.read_parquet."""
    with patch("nested_pandas.read_parquet", wraps=__import__("nested_pandas").read_parquet) as mock_read:
        _ = NestedPandasDataset(
            config={
                "data_set": {"NestedPandasDataset": {"read_parquet_kwargs": {"use_pandas_metadata": False}}}
            },
            data_location=nested_parquet_file,
        )

    mock_read.assert_called_once_with(
        str(nested_parquet_file),
        use_pandas_metadata=False,
    )


def test_nested_pandas_dataset_requires_data_location():
    """Verify the dataset fails clearly when data_location is omitted."""
    with pytest.raises(ValueError):
        NestedPandasDataset(config={"data_set": {"NestedPandasDataset": {"read_parquet_kwargs": {}}}})
