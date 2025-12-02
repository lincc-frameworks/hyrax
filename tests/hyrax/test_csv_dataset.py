from pathlib import Path

import pytest

import hyrax


@pytest.fixture(scope="function")
def test_hyrax_csv_dataset():
    """Fixture that gives a hyrax object configured to use a simple CSV dataset"""
    this_file_dir = Path(__file__).absolute().parent
    csv_file = this_file_dir / "test_data" / "csv_test" / "sample_data.csv"

    h = hyrax.Hyrax()
    h.config["model_inputs"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxCSVDataset",
                "data_location": str(csv_file),
                "primary_id_field": "object_id",
            },
        },
    }

    return h


def test_csv_dataset_initialization(test_hyrax_csv_dataset):
    """Check that the CSV dataset is correctly initialized and can be accessed"""
    dataset = test_hyrax_csv_dataset.prepare()

    # Dataset has correct length
    assert len(dataset["train"]) == 5


def test_csv_dataset_column_getters(test_hyrax_csv_dataset):
    """Check that column getter methods are dynamically created"""
    dataset = test_hyrax_csv_dataset.prepare()

    # Get the underlying HyraxCSVDataset instance
    csv_dataset = dataset["train"]._primary_or_first_dataset()

    # Check that getter methods exist for each column
    assert hasattr(csv_dataset, "get_object_id")
    assert hasattr(csv_dataset, "get_ra")
    assert hasattr(csv_dataset, "get_dec")
    assert hasattr(csv_dataset, "get_magnitude")
    assert hasattr(csv_dataset, "get_flux")
    assert hasattr(csv_dataset, "get_classification")

    # Check that getter methods return correct values
    assert csv_dataset.get_object_id(0) == 1001
    assert csv_dataset.get_ra(0) == 30.5
    assert csv_dataset.get_classification(0) == "star"


def test_csv_dataset_sample_data(test_hyrax_csv_dataset):
    """Check that sample_data returns the first row correctly"""
    dataset = test_hyrax_csv_dataset.prepare()

    # Get the underlying HyraxCSVDataset instance
    csv_dataset = dataset["train"]._primary_or_first_dataset()
    sample = csv_dataset.sample_data()

    # Check that sample has the expected structure
    assert "data" in sample
    assert "object_id" in sample["data"]
    assert "ra" in sample["data"]
    assert "dec" in sample["data"]
    assert "magnitude" in sample["data"]
    assert "flux" in sample["data"]
    assert "classification" in sample["data"]

    # Check that sample values match the first row
    assert sample["data"]["object_id"] == 1001
    assert sample["data"]["ra"] == 30.5
    assert sample["data"]["classification"] == "star"
