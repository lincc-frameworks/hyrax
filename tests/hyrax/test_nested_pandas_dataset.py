from pathlib import Path

import nested_pandas as npd
import pandas as pd
import pytest

from hyrax.datasets.nested_pandas_dataset import NestedPandasDataset


def _write_nested_lightcurve_parquet(path: Path) -> None:
    base = npd.NestedFrame({"object_id": ["o1", "o2", "o3"]})
    nested_rows = [
        pd.DataFrame({"time": [1.0, 2.0], "flux": [10.0, 11.0]}),
        pd.DataFrame({"time": [3.0, 4.0], "flux": [12.0, 13.0]}),
        pd.DataFrame({"time": [5.0], "flux": [14.0]}),
    ]
    nested_frame = base.join_nested(nested_rows, name="lightcurve")
    nested_frame.to_parquet(path)


def test_nested_pandas_dataset_reads_parquet_and_exposes_nested_fields(tmp_path):
    data_path = tmp_path / "random_lightcurve.parquet"
    _write_nested_lightcurve_parquet(data_path)

    dataset = NestedPandasDataset(
        config={"data_set": {}},
        data_location=data_path,
    )

    assert len(dataset) == 3
    assert dataset.get_object_id(0) == "o1"
    assert getattr(dataset, "get_lightcurve.time")(0).tolist() == [1.0, 2.0]
    assert getattr(dataset, "get_lightcurve.flux")(1).tolist() == [12.0, 13.0]


def test_nested_pandas_dataset_passes_read_kwargs_and_max_samples(tmp_path):
    data_path = tmp_path / "random_lightcurve.parquet"
    _write_nested_lightcurve_parquet(data_path)

    dataset = NestedPandasDataset(
        config={
            "data_set": {
                "NestedPandasDataset": {
                    "read_kwargs": {"columns": ["object_id", "lightcurve"]},
                    "max_samples": 2,
                }
            }
        },
        data_location=data_path,
    )

    assert len(dataset) == 2
    assert dataset.get_object_id(1) == "o2"
    assert getattr(dataset, "get_lightcurve.time")(1).tolist() == [3.0, 4.0]


def test_nested_pandas_dataset_requires_data_location():
    with pytest.raises(ValueError):
        NestedPandasDataset(config={"data_set": {}})
