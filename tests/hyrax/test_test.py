import pytest

from hyrax.config_utils import find_most_recent_results_dir


@pytest.fixture(scope="function")
def loopback_hyrax_map_only(tmp_path_factory):
    """This generates a loopback hyrax instance with map-style datasets only,
    suitable for testing the test verb which requires explicit test dataset."""
    import hyrax

    results_dir = tmp_path_factory.mktemp("loopback_hyrax_test")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(results_dir)
    h.config["general"]["dev_mode"] = True

    h.config["model_inputs"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "validate": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "test": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data_infer")),
                "primary_id_field": "object_id",
            },
        },
    }
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    weights_file = results_dir / "fakeweights"
    with open(weights_file, "a"):
        pass
    h.config["infer"]["model_weights_file"] = str(weights_file)

    dataset = h.prepare()
    return h, dataset


def test_test(loopback_hyrax_map_only):
    """
    Simple test that testing succeeds with the loopback
    model in use.
    """
    h, _ = loopback_hyrax_map_only
    # First train a model to have weights to test
    h.train()

    # Now test the model
    metrics = h.test()

    # Verify we got metrics back
    assert metrics is not None
    assert "avg_loss" in metrics


def test_test_with_explicit_weights(loopback_hyrax_map_only, tmp_path):
    """
    Ensure that testing works when explicitly providing model weights file.
    """
    h, _ = loopback_hyrax_map_only
    # set results directory to a temporary path
    h.config["general"]["results_dir"] = str(tmp_path)

    # First, run training to create a saved model file
    _ = h.train()

    # find the model file in the most recent results directory
    results_dir = find_most_recent_results_dir(h.config, "train")
    weights_path = results_dir / h.config["train"]["weights_filename"]

    # Now, set the test config to point to this weights file
    h.config["test"]["model_weights_file"] = str(weights_path)

    # Run test
    metrics = h.test()

    # Verify we got metrics back
    assert metrics is not None
    assert "avg_loss" in metrics


def test_test_auto_detects_weights(loopback_hyrax_map_only, tmp_path):
    """
    Ensure that testing can auto-detect weights from the most recent train run
    when no model_weights_file is specified.
    """
    h, _ = loopback_hyrax_map_only
    # set results directory to a temporary path
    h.config["general"]["results_dir"] = str(tmp_path)

    # First, run training to create a saved model file
    _ = h.train()

    # Ensure test config doesn't have weights file specified
    h.config["test"] = {"model_weights_file": False}

    # Run test - should auto-detect weights from train
    metrics = h.test()

    # Verify we got metrics back
    assert metrics is not None
    assert "avg_loss" in metrics


def test_test_saves_weights_file(loopback_hyrax_map_only, tmp_path):
    """
    Ensure that testing saves the model weights to test_weights.pth.
    """
    h, _ = loopback_hyrax_map_only
    # set results directory to a temporary path
    h.config["general"]["results_dir"] = str(tmp_path)

    # First, run training to create a saved model file
    _ = h.train()

    # Run test
    _ = h.test()

    # Find the most recent test results directory
    test_results_dir = find_most_recent_results_dir(h.config, "test")

    # Verify that test_weights.pth was saved
    weights_file = test_results_dir / "test_weights.pth"
    assert weights_file.exists(), f"Expected weights file at {weights_file} does not exist"
