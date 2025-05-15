import numpy as np
import pytest

import hyrax


@pytest.fixture(scope="function", params=["RandomDataset", "RandomIterableDataset"])
def loopback_hyrax(tmp_path_factory, request):
    """This generates a loopback hyrax instance
    which is configured to use the loopback model
    and a simple dataset yielding random numbers
    """
    results_dir = tmp_path_factory.mktemp(f"loopback_hyrax_{request.param}")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "LoopbackModel"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(results_dir)

    h.config["general"]["dev_mode"] = True
    h.config["data_set"]["name"] = request.param
    h.config["data_set"]["size"] = 20
    h.config["data_set"]["seed"] = 0

    h.config["data_set"]["validate_size"] = 0.2
    h.config["data_set"]["test_size"] = 0.2
    h.config["data_set"]["train_size"] = 0.6

    weights_file = results_dir / "fakeweights"
    with open(weights_file, "a"):
        pass
    h.config["infer"]["model_weights_file"] = str(weights_file)

    dataset = h.prepare()
    return h, dataset


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
        assert all(np.isclose(dataset[dataset_idx], inference_results[idx]))
