import pandas as pd
import pytest

from hyrax.datasets.nested_pandas_dataset import NestedPandasDataset


def _config(read_parquet_kwargs=None):
    return {
        "data_set": {
            "NestedPandasDataset": {
                "read_parquet_kwargs": read_parquet_kwargs or {},
            }
        }
    }


@pytest.fixture
def sample_parquet(tmp_path):
    """Write a 3-row nested-pandas parquet with flat columns (object_id, ra) and nested column (lc)."""
    import nested_pandas as npd

    flat = npd.NestedFrame(
        {
            "object_id": ["obj_0", "obj_1", "obj_2"],
            "ra": [10.0, 20.0, 30.0],
        }
    )
    nested = pd.DataFrame(
        {
            "time": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "flux": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "object_id": ["obj_0", "obj_0", "obj_1", "obj_1", "obj_2", "obj_2"],
        }
    ).set_index("object_id")
    nf = flat.set_index("object_id").join_nested(nested, "lc")

    path = tmp_path / "sample.parquet"
    nf.to_parquet(str(path))
    return path


def test_dataset_length(sample_parquet):
    dataset = NestedPandasDataset(config=_config(), data_location=sample_parquet)
    assert len(dataset) == 3


def test_flat_getters(sample_parquet):
    dataset = NestedPandasDataset(config=_config(), data_location=sample_parquet)
    assert dataset.get_object_id(0) == "obj_0"
    assert dataset.get_object_id(2) == "obj_2"
    assert dataset.get_ra(1) == pytest.approx(20.0)


def test_nested_getter_returns_dataframe(sample_parquet):
    dataset = NestedPandasDataset(config=_config(), data_location=sample_parquet)
    lc = dataset.get_lc(0)
    assert isinstance(lc, pd.DataFrame)
    assert list(lc.columns) == ["time", "flux"]
    assert len(lc) == 2
    assert lc["flux"].iloc[0] == pytest.approx(1.0)


def test_requires_data_location():
    with pytest.raises(ValueError):
        NestedPandasDataset(config=_config())


def test_read_parquet_kwargs_forwarded(sample_parquet, monkeypatch):
    import nested_pandas as npd

    original_read = npd.read_parquet
    captured = {}

    def wrapped(data, **kwargs):
        captured.update(kwargs)
        return original_read(data, **kwargs)

    monkeypatch.setattr(npd, "read_parquet", wrapped)

    NestedPandasDataset(
        config=_config(read_parquet_kwargs={"reject_nesting": []}),
        data_location=sample_parquet,
    )

    assert "reject_nesting" in captured
    assert captured["reject_nesting"] == []
