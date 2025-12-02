from hyrax.config_utils import find_most_recent_results_dir


def test_train(loopback_hyrax):
    """
    Simple test that training succeeds with the loopback
    model in use.
    """
    h, _ = loopback_hyrax
    h.train()


def test_train_resume(loopback_hyrax, tmp_path):
    """
    Ensure that training can be resumed from a checkpoint
    when using the loopback model.
    """
    checkpoint_filename = "checkpoint_epoch_1.pt"

    h, _ = loopback_hyrax
    # set results directory to a temporary path
    h.config["general"]["results_dir"] = str(tmp_path)

    # First, run initial training to create a saved model file
    _ = h.train()

    # find the model file in the most recent results directory
    results_dir = find_most_recent_results_dir(h.config, "train")
    checkpoint_path = results_dir / checkpoint_filename

    # Now, set the resume config to point to this checkpoint
    h.config["train"]["resume"] = str(checkpoint_path)

    # Resume training
    h.train()


def test_train_percent_split(tmp_path):
    """
    Ensure backward compatibility with percent-based splits when the
    configuration provides only a `train` and `infer` model_inputs section
    (no explicit `validate` table). This should exercise the code path
    that creates train/validate splits from a single dataset location.
    """
    import hyrax

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 4
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    # Only provide `train` and `infer` model_inputs (no `validate` key).
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
    # location `train` dataset and produce a validate split implicitly.
    h.config["data_set"]["train_size"] = 0.6
    h.config["data_set"]["validate_size"] = 0.2
    h.config["data_set"]["test_size"] = 0.2

    # Instead of running full training, validate that the legacy percent-based
    # split path creates both train and validate dataloaders with expected sizes.
    from hyrax.pytorch_ignite import dist_data_loader, setup_dataset

    # Create dataset dict using the same logic as training
    dataset = setup_dataset(h.config)

    assert "train" in dataset

    data_loaders = dist_data_loader(dataset["train"], h.config, ["train", "validate"])

    # Should have created both train and validate loaders
    assert "train" in data_loaders and "validate" in data_loaders

    train_loader, train_indexes = data_loaders["train"]
    validate_loader, validate_indexes = data_loaders["validate"]

    # Assert expected sizes: train 12 (60% of 20), validate 4 (20% of 20)
    assert len(train_indexes) == 12
    assert len(validate_indexes) == 4

    # Finally, run full training to exercise `train.py` end-to-end and ensure
    # the training verb functions correctly with percent-based splits.
    h.train()
