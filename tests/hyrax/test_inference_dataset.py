import numpy as np
import pytest

import hyrax
from hyrax.datasets.data_provider import DataProvider
from hyrax.datasets.inference_dataset import InferenceDataset, InferenceDatasetWriter


@pytest.fixture(scope="session", params=[1, 2, 3, 4, 5])
def inference_dataset(tmp_path_factory, request):
    """Fixture where I write test data in an InferenceDatasetWriter
    It returns the data written"""
    h = hyrax.Hyrax()
    h.config["general"]["dev_mode"] = True
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
    }
    h.config["split"] = {"train": 1.0}
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2]
    original_data_set = h.prepare()

    current_data_set = original_data_set["train"]

    for round_number in range(request.param):
        tmp_path = tmp_path_factory.mktemp(f"order_test_{request.param}_{round_number}")

        data_writer = InferenceDatasetWriter(current_data_set, tmp_path)

        indexes = np.array(range(20))
        np.random.shuffle(indexes)

        # On the first iteration, we get a DataProvider, but on subsequent iterations we have
        # an ResultDataset; this method of enumeration always works.
        data_set_ids = np.array(current_data_set.ids())

        data_writer.write_batch(
            np.array(data_set_ids[indexes[0:10]]),  # ids
            get_data_by_dataset_type(current_data_set, indexes[0:10]),  # Results
        )

        data_writer.write_batch(
            np.array(data_set_ids[indexes[10:20]]),  # ids
            get_data_by_dataset_type(current_data_set, indexes[10:20]),  # Results
        )
        data_writer.write_index()
        current_data_set = InferenceDataset(h.config, tmp_path)

    return original_data_set["train"], current_data_set


def get_data_by_dataset_type(dataset, idxs):
    """Different behavior depending on whether the dataset is an `InferenceDataset`
    vs. a DataProvider dataset. IF it's an InferenceDataset, we return the data
    directly, otherwise we unpack the data from the DataProvider."""

    def _get_data_by_dataset_type(dataset, idx):
        output_data = dataset[idx]
        if isinstance(dataset, DataProvider):
            output_data = output_data["data"]["image"]

        return np.array(output_data)

    return [_get_data_by_dataset_type(dataset, int(i)) for i in idxs]


def test_order(inference_dataset):
    """Test ID ordering consistency between original and inference datasets.

    Test cases:
    1) ids() should not be in the same order between original and result
    2) ids() should contain all the IDs in the original dataset
    3) The value from inference_dataset[idx] should match a value from data_set
    3a) The matching values should have the same ID in both inference_dataset and data_set according to .ids()
    """
    orig, result = inference_dataset

    orig_ids = orig.ids()
    result_ids = result.ids()

    # Check no IDs are dropped
    for id in orig_ids:
        assert id in result_ids

    # Check all data is the correct data for that ID
    for result_i in range(20):
        for orig_i in range(20):
            if np.all(orig[orig_i]["data"]["image"] == result[result_i].numpy()):
                assert orig_ids[orig_i] == result_ids[result_i]
                break
        else:
            assert False, "Could not find matching value for ID."  # noqa: B011
