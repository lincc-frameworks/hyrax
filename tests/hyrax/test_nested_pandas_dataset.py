import pandas as pd
import pytest

from hyrax.datasets.nested_pandas_dataset import HyraxNestedPandasDataset


@pytest.fixture
def sample_nested_data(tmp_path):
    """Create minimal nested-pandas data on disk and return the path."""
    import nested_pandas as npd

    # Create main object catalog
    catalog = pd.DataFrame(
        {
            "object_id": ["obj1", "obj2", "obj3"],
            "magnitude": [15.5, 16.2, 14.8],
            "classification": ["star", "galaxy", "star"],
        }
    )

    # Create nested measurements dataframes grouped by object
    measurements_full = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
            "flux": [100.0, 200.0, 150.0, 110.0, 210.0, 160.0, 120.0, 220.0, 170.0],
            "error": [5.0, 10.0, 8.0, 5.5, 10.5, 8.5, 6.0, 11.0, 9.0],
        }
    )

    # Group measurements by object_id
    all_measurements = [
        measurements_full.iloc[0:3],  # first 3 rows for obj1
        measurements_full.iloc[3:6],  # next 3 rows for obj2
        measurements_full.iloc[6:9],  # last 3 rows for obj3
    ]

    catalog["measurements"] = all_measurements

    # Convert to nested-pandas dataframe
    ndf = npd.NestedFrame(catalog)

    # Save using pickle (nested dataframes cannot be saved to parquet directly)
    data_path = tmp_path / "sample_data.pkl"
    ndf.to_pickle(data_path)

    return data_path


def test_nested_pandas_dataset_length(sample_nested_data):
    """Test that dataset length matches the number of top-level objects."""
    dataset = HyraxNestedPandasDataset(
        config={
            "data_set": {
                "HyraxNestedPandasDataset": {
                    "read_kwargs": {},
                }
            }
        },
        data_location=sample_nested_data,
    )

    assert len(dataset) == 3


def test_nested_pandas_dataset_top_level_getters(sample_nested_data):
    """Test that top-level column getters work correctly."""
    dataset = HyraxNestedPandasDataset(
        config={
            "data_set": {
                "HyraxNestedPandasDataset": {
                    "read_kwargs": {},
                }
            }
        },
        data_location=sample_nested_data,
    )

    # Test top-level getters
    assert dataset.get_object_id(0) == "obj1"
    assert dataset.get_magnitude(1) == pytest.approx(16.2)
    assert dataset.get_classification(2) == "star"


def test_nested_pandas_dataset_nested_getters(sample_nested_data):
    """Test that nested dataframe column getters work correctly."""
    dataset = HyraxNestedPandasDataset(
        config={
            "data_set": {
                "HyraxNestedPandasDataset": {
                    "read_kwargs": {},
                }
            }
        },
        data_location=sample_nested_data,
    )

    # Test nested getters - they should return arrays
    times_0 = dataset.get_measurements_time(0)
    assert len(times_0) == 3
    assert times_0[0] == pytest.approx(1.0)
    assert times_0[1] == pytest.approx(2.0)

    flux_1 = dataset.get_measurements_flux(1)
    assert len(flux_1) == 3
    assert flux_1[0] == pytest.approx(110.0)

    errors_2 = dataset.get_measurements_error(2)
    assert len(errors_2) == 3
    assert errors_2[1] == pytest.approx(11.0)


def test_nested_pandas_dataset_missing_data_location():
    """Test that missing data_location raises ValueError."""
    with pytest.raises(ValueError, match="data_location"):
        HyraxNestedPandasDataset(
            config={
                "data_set": {
                    "HyraxNestedPandasDataset": {
                        "read_kwargs": {},
                    }
                }
            },
            data_location=None,
        )


def test_nested_pandas_dataset_getitem(sample_nested_data):
    """Test that __getitem__ returns an empty dict."""
    dataset = HyraxNestedPandasDataset(
        config={
            "data_set": {
                "HyraxNestedPandasDataset": {
                    "read_kwargs": {},
                }
            }
        },
        data_location=sample_nested_data,
    )

    result = dataset[0]
    assert result == {}


def test_nested_pandas_dataset_sample_data(sample_nested_data):
    """Test that sample_data returns the first row correctly."""
    dataset = HyraxNestedPandasDataset(
        config={
            "data_set": {
                "HyraxNestedPandasDataset": {
                    "read_kwargs": {},
                }
            }
        },
        data_location=sample_nested_data,
    )

    sample = dataset.sample_data()

    # Check structure
    assert "data" in sample
    assert "object_id" in sample["data"]
    assert "magnitude" in sample["data"]
    assert "classification" in sample["data"]
    assert "measurements" in sample["data"]

    # Check values
    assert sample["data"]["object_id"] == "obj1"
    assert sample["data"]["magnitude"] == pytest.approx(15.5)
    assert sample["data"]["classification"] == "star"
