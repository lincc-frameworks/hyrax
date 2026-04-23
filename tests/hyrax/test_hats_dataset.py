from pathlib import Path

import pandas as pd
import pytest

import hyrax

lsdb = pytest.importorskip("lsdb")


@pytest.fixture(scope="function")
def hats_catalog_path(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "object_id": [1001, 1002, 1003],
            "coord_ra": [150.1, 150.2, 150.3],
            "coord_dec": [2.1, 2.2, 2.3],
            "mag-r": [19.1, 19.4, 18.9],
        }
    )
    catalog = lsdb.from_dataframe(df, ra_column="coord_ra", dec_column="coord_dec")

    out_path = tmp_path / "sample_hats"
    catalog.write_catalog(out_path)

    return out_path


@pytest.fixture(scope="function")
def test_hyrax_hats_dataset(hats_catalog_path: Path):
    h = hyrax.Hyrax()
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxHATSDataset",
                "data_location": str(hats_catalog_path),
                "primary_id_field": "object_id",
                "split_fraction": 1.0,
            }
        }
    }
    return h


def test_hats_dataset_initialization(test_hyrax_hats_dataset):
    dataset = test_hyrax_hats_dataset.prepare()
    assert len(dataset["train"]) == 3


def test_hats_dataset_dynamic_getters(test_hyrax_hats_dataset):
    dataset = test_hyrax_hats_dataset.prepare()
    hats_dataset = dataset["train"]._primary_or_first_dataset()

    assert hasattr(hats_dataset, "get_object_id")
    assert hasattr(hats_dataset, "get_coord_ra")
    assert hasattr(hats_dataset, "get_coord_dec")
    assert hasattr(hats_dataset, "get_mag-r")

    assert hats_dataset.get_object_id(0) == 1001
    assert hats_dataset.get_coord_ra(1) == pytest.approx(150.2)
    assert getattr(hats_dataset, "get_mag-r")(2) == pytest.approx(18.9)


def test_hats_dataset_sample_data(test_hyrax_hats_dataset):
    dataset = test_hyrax_hats_dataset.prepare()
    hats_dataset = dataset["train"]._primary_or_first_dataset()
    sample = hats_dataset.sample_data()

    assert "data" in sample
    assert sample["data"]["object_id"] == 1001
    assert sample["data"]["coord_ra"] == pytest.approx(150.1)
    assert sample["data"]["mag-r"] == pytest.approx(19.1)


def test_hats_dataset_with_data_request_fields_only_builds_requested_getters(hats_catalog_path: Path):
    h = hyrax.Hyrax()
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxHATSDataset",
                "data_location": str(hats_catalog_path),
                "fields": ["coord_ra"],
                "primary_id_field": "object_id",
                "split_fraction": 1.0,
            }
        }
    }

    dataset = h.prepare()
    hats_dataset = dataset["train"]._primary_or_first_dataset()

    assert hasattr(hats_dataset, "get_object_id")
    assert hasattr(hats_dataset, "get_coord_ra")
    assert not hasattr(hats_dataset, "get_coord_dec")
    assert not hasattr(hats_dataset, "get_mag-r")


def test_hats_dataset_open_catalog_filters_from_dataset_config(hats_catalog_path: Path):
    h = hyrax.Hyrax()
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxHATSDataset",
                "data_location": str(hats_catalog_path),
                "fields": ["coord_ra"],
                "primary_id_field": "object_id",
                "split_fraction": 1.0,
                "dataset_config": {
                    "HyraxHATSDataset": {
                        "open_catalog_kwargs": {
                            "filters": [("coord_ra", ">", 150.15)],
                        }
                    }
                },
            }
        }
    }

    dataset = h.prepare()
    hats_dataset = dataset["train"]._primary_or_first_dataset()

    assert len(hats_dataset) == 2
    assert hasattr(hats_dataset, "get_object_id")
    assert hasattr(hats_dataset, "get_coord_ra")
    assert not hasattr(hats_dataset, "get_coord_dec")
    assert not hasattr(hats_dataset, "get_mag-r")
    assert hats_dataset.get_object_id(0) == 1002
