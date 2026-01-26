from hyrax.config_utils import find_most_recent_results_dir


def test_test(loopback_hyrax):
    """
    Simple test that testing succeeds with the loopback
    model in use.
    """
    h, _ = loopback_hyrax
    # First train a model to have weights to test
    h.train()
    
    # Now test the model
    metrics = h.test()
    
    # Verify we got metrics back
    assert metrics is not None
    assert "test_loss" in metrics or "avg_loss" in metrics


def test_test_with_explicit_weights(loopback_hyrax, tmp_path):
    """
    Ensure that testing works when explicitly providing model weights file.
    """
    h, _ = loopback_hyrax
    # set results directory to a temporary path
    h.config["general"]["results_dir"] = str(tmp_path)

    # First, run training to create a saved model file
    _ = h.train()

    # find the model file in the most recent results directory
    results_dir = find_most_recent_results_dir(h.config, "train")
    weights_path = results_dir / h.config["train"]["weights_filename"]

    # Now, set the test config to point to this weights file
    h.config["test"] = {"model_weights_file": str(weights_path)}

    # Run test
    metrics = h.test()
    
    # Verify we got metrics back
    assert metrics is not None
    assert "test_loss" in metrics or "avg_loss" in metrics


def test_test_auto_detects_weights(loopback_hyrax, tmp_path):
    """
    Ensure that testing can auto-detect weights from the most recent train run
    when no model_weights_file is specified.
    """
    h, _ = loopback_hyrax
    # set results directory to a temporary path
    h.config["general"]["results_dir"] = str(tmp_path)

    # First, run training to create a saved model file
    _ = h.train()

    # Ensure test config doesn't have weights file specified
    h.config["test"] = {"model_weights_file": False}
    h.config["infer"]["model_weights_file"] = False

    # Run test - should auto-detect weights from train
    metrics = h.test()
    
    # Verify we got metrics back
    assert metrics is not None
    assert "test_loss" in metrics or "avg_loss" in metrics


def test_test_percent_split(tmp_path):
    """
    Ensure that testing works with percent-based splits when the
    configuration provides only a `train` and `infer` model_inputs section
    (no explicit `test` table). This should exercise the code path
    that creates test splits from a single dataset location.
    """
    import hyrax

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 4
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    # Only provide `train` and `infer` model_inputs (no `test` key).
    h.config["model_inputs"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data_train"),
                "primary_id_field": "object_id",
            }
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data_infer"),
                "primary_id_field": "object_id",
            }
        },
    }

    # Configure the underlying random dataset used by tests
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    # Percent-based split parameters - these should be applied to the single
    # location `train` dataset and produce a test split implicitly.
    h.config["data_set"]["train_size"] = 0.6
    h.config["data_set"]["validate_size"] = 0.2
    h.config["data_set"]["test_size"] = 0.2

    # First train a model
    h.train()
    
    # Then test the model - should use the test split from the train dataset
    metrics = h.test()
    
    # Verify we got metrics back
    assert metrics is not None
    assert "test_loss" in metrics or "avg_loss" in metrics
