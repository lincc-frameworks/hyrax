import numpy as np
import pytest

import hyrax
from hyrax.data_sets.data_provider import DataProvider
from hyrax.data_sets.inference_dataset import InferenceDataSet, InferenceDataSetWriter


@pytest.fixture(scope="session", params=[1, 2, 3, 4, 5])
def inference_dataset(tmp_path_factory, request):
    """Fixture where I write test data in an InferenceDataSetWriter
    It returns the data written"""
    h = hyrax.Hyrax()
    h.config["general"]["dev_mode"] = True
    h.config["model_inputs"] = {
        "data": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": str(tmp_path_factory.mktemp("data")),
            "primary_id_field": "object_id",
        },
    }
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2]
    original_data_set = h.prepare()

    current_data_set = original_data_set

    for round_number in range(request.param):
        tmp_path = tmp_path_factory.mktemp(f"order_test_{request.param}_{round_number}")

        data_writer = InferenceDataSetWriter(current_data_set, tmp_path)

        indexes = np.array(range(20))
        np.random.shuffle(indexes)

        data_set_ids = np.array(list(current_data_set.ids()))

        data_writer.write_batch(
            np.array(data_set_ids[indexes[0:10]]),  # ids
            get_data_by_dataset_type(current_data_set, indexes[0:10]),  # Results
        )

        data_writer.write_batch(
            np.array(data_set_ids[indexes[10:20]]),  # ids
            get_data_by_dataset_type(current_data_set, indexes[10:20]),  # Results
        )
        data_writer.write_index()
        current_data_set = InferenceDataSet(h.config, tmp_path)

    return original_data_set, current_data_set


def get_data_by_dataset_type(dataset, idx):
    """Different behavior depending on whether the dataset is an `InferenceDataset`
    vs. a DataProvider dataset. IF it's an InferenceDataset, we return the data
    directly, otherwise we unpack the data from the DataProvider."""
    output_data = dataset[idx]
    if isinstance(dataset, DataProvider):
        output_data = output_data["data"]["image"]

    return np.array(output_data)


def test_order(inference_dataset):
    """Test ID and metadata ordering consistency between original and inference datasets.

    Test cases:
    1) ids() should not be in the same order between original and result
    2) ids() should contain all the IDs in the original dataset
    3) the ids() from ids, and the ids delivered via metadata from the inference dataset should match exactly
    4) The value from inference_dataset[idx] should match a value from data_set
    4a) The matching values should have the same ID in both inference_dataset and data_set according to .ids()
    4b) The matching values should have the same ID in both inference_dataset and data_set according to
        respective metadata
    5) metadata() must preserve the exact order of requested indices (critical for visualization correctness)
    """
    orig, result = inference_dataset

    orig_ids = list(orig.ids())
    result_ids = list(result.ids())

    all_idx = list(range(20))
    orig_meta_ids = list(orig.metadata(all_idx, ["object_id_data"])["object_id_data"])
    result_meta_ids = list(result.metadata(all_idx, ["object_id_data"])["object_id_data"])

    # Check no IDs are dropped
    for id in orig_ids:
        assert id in result_ids
        assert id in orig_meta_ids
        assert id in result_meta_ids

    # Check all data is the correct data for that ID
    for result_i in range(20):
        for orig_i in range(20):
            if np.all(orig[orig_i]["data"]["image"] == result[result_i].numpy()):
                try:
                    assert orig_ids[orig_i] == result_ids[result_i]
                    assert orig_meta_ids[orig_i] == result_meta_ids[result_i]
                    assert orig_ids[orig_i] == result_meta_ids[result_i]
                    assert orig_meta_ids[orig_i] == result_ids[result_i]
                except Exception as e:
                    print(f"Original ID: {orig_ids[orig_i]} (Correct b/c data matches)")
                    print(f"Original metaID: {orig_meta_ids[orig_i]}")
                    print(f"Result ID: {result_ids[result_i]}")
                    print(f"Result metaID: {result_meta_ids[result_i]}")
                    raise e
                break
        else:
            assert False, "Could not find matching value for ID."  # noqa: B011

    # Test explicit metadata ordering preservation with non-sequential patterns
    test_patterns = [
        [3, 1, 4, 0, 2],
        [19, 5, 10, 15, 2],
        [0, 19, 1, 18],
    ]

    for idx_pattern in test_patterns:
        # Skip patterns that exceed dataset size
        if max(idx_pattern) >= len(result):
            continue

        metadata_result = result.metadata(idx_pattern, ["object_id_data"])

        # Get expected IDs in the exact order requested
        expected_ids = [list(result.ids())[i] for i in idx_pattern]
        actual_ids = [str(id) for id in metadata_result["object_id_data"]]

        assert actual_ids == expected_ids, (
            f"CRITICAL: Metadata ordering broken! For indices {idx_pattern}:\n"
            f"Expected IDs: {expected_ids}\n"
            f"Actual IDs:   {actual_ids}\n"
            f"This will cause visualization labels and data to be scrambled."
        )
