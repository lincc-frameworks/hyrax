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


def test_train_split_fraction(tmp_path):
    """
    Test training with split_fraction on groups sharing the same data_location.
    This should exercise Path 2 where setup_dataset assigns split_indices to
    each DataProvider, and the train verb creates dataloaders with those
    split_indices applied via SubsetSequentialSampler.
    """
    import hyrax

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 4
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    # Define train and validate groups pointing to the SAME data_location
    # with split_fraction set on each.
    shared_location = str(tmp_path / "shared_data")
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": shared_location,
                "primary_id_field": "object_id",
                "split_fraction": 0.7,
            }
        },
        "validate": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": shared_location,
                "primary_id_field": "object_id",
                "split_fraction": 0.3,
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

    # Configure the underlying random dataset
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 30
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 42
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    # Run training - this should use Path 2 (split_fraction)
    model = h.train()

    # Verify training completed
    assert model is not None

    # Additional verification: check that split_indices were properly set
    from hyrax.pytorch_ignite import setup_dataset

    dataset = setup_dataset(h.config, splits=("train", "validate"))

    # Both train and validate should have split_indices
    assert hasattr(dataset["train"], "split_indices")
    assert hasattr(dataset["validate"], "split_indices")
    assert dataset["train"].split_indices is not None
    assert dataset["validate"].split_indices is not None

    # Verify expected sizes: 70% of 30 = 21, 30% of 30 = 9
    assert len(dataset["train"].split_indices) == 21
    assert len(dataset["validate"].split_indices) == 9

    # Verify indices are non-overlapping
    train_set = set(dataset["train"].split_indices)
    validate_set = set(dataset["validate"].split_indices)
    assert len(train_set & validate_set) == 0

    # Verify indices cover the full range
    assert train_set | validate_set == set(range(30))


def test_constant_scheduler(loopback_hyrax):
    """
    Ensure that setting a ConstantLR works properly
    """
    h, _ = loopback_hyrax
    factor = 0.5
    h.config["scheduler"]["name"] = "torch.optim.lr_scheduler.ConstantLR"
    h.config["torch.optim.lr_scheduler.ConstantLR"] = {"total_iters": 4, "factor": factor}
    h.config["train"]["epochs"] = 6
    initial_lr = 128
    h.config[h.config["optimizer"]["name"]]["lr"] = 128
    model = h.train()

    assert hasattr(model, "_learning_rates_history")
    assert model._learning_rates_history == [[initial_lr * factor]] * 4 + [[initial_lr]] * 2


def test_exponential_scheduler(loopback_hyrax):
    """
    Ensure that setting an ExponentialLR scheduler works properly
    """
    h, _ = loopback_hyrax
    gamma = 0.5
    h.config["scheduler"]["name"] = "torch.optim.lr_scheduler.ExponentialLR"
    h.config["torch.optim.lr_scheduler.ExponentialLR"] = {"gamma": gamma}
    h.config["train"]["epochs"] = 5
    initial_lr = 128
    h.config[h.config["optimizer"]["name"]]["lr"] = initial_lr
    model = h.train()

    assert hasattr(model, "_learning_rates_history")
    assert model._learning_rates_history == [
        [initial_lr * gamma**i] for i in range(h.config["train"]["epochs"])
    ]


def test_exponential_scheduler_checkpointing(loopback_hyrax, tmp_path):
    """
    Ensure that ExponentialLR scheduler resumes from a checkpoint properly
    """
    checkpoint_filename = "checkpoint_epoch_3.pt"
    h, _ = loopback_hyrax

    # set results directory to a temporary path
    h.config["general"]["results_dir"] = str(tmp_path)

    # set the scheduler up
    gamma = 0.5
    initial_lr = 128
    h.config["scheduler"]["name"] = "torch.optim.lr_scheduler.ExponentialLR"
    h.config["torch.optim.lr_scheduler.ExponentialLR"] = {"gamma": gamma}
    h.config["train"]["epochs"] = 3
    h.config[h.config["optimizer"]["name"]]["lr"] = initial_lr

    # run initial training to create a saved model file
    model = h.train()

    # first 3 epochs working as expected
    assert hasattr(model, "_learning_rates_history")
    assert model._learning_rates_history == [[initial_lr * gamma**i] for i in range(3)]

    # find the model file in the most recent results directory
    results_dir = find_most_recent_results_dir(h.config, "train")
    checkpoint_path = results_dir / checkpoint_filename

    # Now, set the resume config to point to this checkpoint
    h.config["train"]["resume"] = str(checkpoint_path)

    # We will try running for two more epochs
    h.config["train"]["epochs"] = 5
    # Resume training
    model = h.train()

    assert hasattr(model, "_learning_rates_history")
    assert model._learning_rates_history == [[initial_lr * gamma**i] for i in range(3, 5)]


def test_constant_scheduler_checkpointing(loopback_hyrax, tmp_path):
    """
    Ensure that ConstantLR scheduler resumes from a checkpoint properly
    """
    checkpoint_filename = "checkpoint_epoch_2.pt"
    h, _ = loopback_hyrax

    # set results directory to a temporary path
    h.config["general"]["results_dir"] = str(tmp_path)

    # set the scheduler up
    factor = 0.5
    initial_lr = 128
    h.config["scheduler"]["name"] = "torch.optim.lr_scheduler.ConstantLR"
    h.config["torch.optim.lr_scheduler.ConstantLR"] = {"total_iters": 3, "factor": factor}
    h.config["train"]["epochs"] = 2
    h.config[h.config["optimizer"]["name"]]["lr"] = initial_lr

    # run initial training to create a saved model file
    model = h.train()

    # first 2 epochs working as expected
    assert hasattr(model, "_learning_rates_history")
    assert model._learning_rates_history == [[initial_lr * factor]] * h.config["train"]["epochs"]

    # find the model file in the most recent results directory
    results_dir = find_most_recent_results_dir(h.config, "train")
    checkpoint_path = results_dir / checkpoint_filename

    # Now, set the resume config to point to this checkpoint
    h.config["train"]["resume"] = str(checkpoint_path)

    # We will try running for three more epochs
    h.config["train"]["epochs"] = 5
    # Resume training
    model = h.train()

    assert hasattr(model, "_learning_rates_history")
    assert model._learning_rates_history == [[initial_lr * factor]] + [[initial_lr]] * 2
