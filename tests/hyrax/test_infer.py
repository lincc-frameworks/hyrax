import numpy as np
import pytest


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("split", ["test", "train", "validate", None])
def test_infer_order(loopback_hyrax, split, shuffle):
    """Test that the order of data run through infer
    is correct in the presence of several splits
    """
    h, dataset = loopback_hyrax
    h.config["infer"]["split"] = split if split is not None else False
    h.config["data_loader"]["shuffle"] = shuffle

    inference_results = h.infer()
    inference_result_ids = list(inference_results.ids())
    original_dataset_ids = list(dataset.ids())

    if dataset.is_iterable():
        dataset = list(dataset)
        original_dataset_ids = np.array([str(s["object_id"]) for s in dataset])

    for idx, result_id in enumerate(inference_result_ids):
        dataset_idx = None
        for i, orig_id in enumerate(original_dataset_ids):
            if orig_id == result_id:
                dataset_idx = i
                break
        else:
            raise AssertionError("Failed to find a corresponding ID")

        print(f"orig idx: {dataset_idx}, infer idx: {idx}")
        print(f"orig data: {dataset[dataset_idx]}, infer data: {inference_results[idx]}")
        assert np.all(np.isclose(dataset[dataset_idx]["data"]["image"], inference_results[idx]))
