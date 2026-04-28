from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from hyrax.datasets.nested_pandas_dataset import NestedPandasDataset


@pytest.fixture
def nested_pandas_fixture(tmp_path):
    """Create a tiny nested-pandas parquet file with base and nested columns."""
    import nested_pandas as npd

    flat = pd.DataFrame(
        {
            "object_id": ["obj_1", "obj_2", "obj_3"],
            "ra": [10.1, 10.2, 10.3],
            "time": [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0]],
            "flux": [[11.0, 12.0], [21.0, 22.0, 23.0], [31.0]],
        }
    )
    table = npd.NestedFrame.from_lists(
        flat,
        base_columns=["object_id", "ra"],
        list_columns=["time", "flux"],
        name="lightcurve",
    )

    data_path = tmp_path / "nested_sources.parquet"
    table.to_parquet(data_path)
    return data_path


def test_nested_pandas_dataset_reads_file_and_exposes_getters(nested_pandas_fixture):
    """Verify base, nested top-level, and nested subcolumn getters."""
    dataset = NestedPandasDataset(
        config={"data_set": {"NestedPandasDataset": {"read_parquet_kwargs": {}}}},
        data_location=nested_pandas_fixture,
    )

    assert len(dataset) == 3
    assert dataset.get_object_id(0) == "obj_1"
    assert dataset.get_ra(1) == pytest.approx(10.2)
    assert dataset.get_lightcurve(0).to_dict(orient="list") == {
        "time": [1.0, 2.0],
        "flux": [11.0, 12.0],
    }
    np.testing.assert_array_equal(getattr(dataset, "get_lightcurve.time")(1), np.array([3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(getattr(dataset, "get_lightcurve.flux")(2), np.array([31.0]))


def test_nested_pandas_dataset_passes_through_read_parquet_kwargs(monkeypatch, tmp_path):
    """Verify read_parquet_kwargs are passed directly to nested_pandas.read_parquet."""
    import nested_pandas as npd

    mock_table = MagicMock()
    mock_table.columns = ["object_id"]
    mock_table.nested_columns = []
    monkeypatch.setattr(npd, "read_parquet", MagicMock(return_value=mock_table))

    NestedPandasDataset(
        config={
            "data_set": {
                "NestedPandasDataset": {
                    "read_parquet_kwargs": {
                        "columns": ["object_id"],
                        "use_pandas_metadata": False,
                    }
                }
            }
        },
        data_location=tmp_path / "sample.parquet",
    )

    npd.read_parquet.assert_called_once_with(
        str(tmp_path / "sample.parquet"),
        columns=["object_id"],
        use_pandas_metadata=False,
    )


def test_nested_pandas_dataset_requires_data_location():
    """Verify missing data_location fails before attempting to read data."""
    with pytest.raises(ValueError):
        NestedPandasDataset(config={"data_set": {"NestedPandasDataset": {"read_parquet_kwargs": {}}}})
