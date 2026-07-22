import pytest

from hyrax.config_utils import find_most_recent_results_dir


def test_train_trace(loopback_hyrax):
    """
    Integration test: train(trace=N) returns a TraceResult that can be
    printed and inspected stage by stage, modelling how a user would use
    the trace feature from a notebook.
    """
    from hyrax.trace import TraceResult, TraceStage

    h, _ = loopback_hyrax
    trace_result = h.train(trace=4)

    # User would first print the result
    assert isinstance(trace_result, TraceResult)
    assert len(str(trace_result)) > 0

    # User accesses stages via attribute notation: trace_result.evaluation
    assert isinstance(trace_result.evaluation, TraceStage)

    # User accesses stages via dict notation: trace_result["collate"]
    assert isinstance(trace_result["collate"], TraceStage)

    # Stages should have captured calls from the data pipeline
    assert len(trace_result["evaluation"]) > 0
    assert len(trace_result["collate"]) > 0

    # User can dive into individual calls within a stage
    first_eval_call = trace_result["evaluation"][0]
    assert str(first_eval_call)

    # Verify access to captured function parameters by name and by number
    # The evaluation stage captures {"batch": 0} as the first positional parameter
    batch_by_name = first_eval_call["batch"]
    assert batch_by_name is not None

    batch_by_number = first_eval_call[0]
    assert batch_by_number is not None

    # Numeric and named access should return the same captured parameter value
    assert batch_by_name is batch_by_number


def test_train(loopback_hyrax):
    """
    Simple test that training succeeds with the loopback
    model in use.
    """
    h, _ = loopback_hyrax
    h.train()


def test_best_checkpoint_uses_validation_loss(loopback_hyrax, tmp_path):
    """
    When a validator is present, the best checkpoint should be scored on
    validation loss (not training loss).  Verify that a best-checkpoint
    file is written to the results directory after training.
    """
    h, _ = loopback_hyrax
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["train"]["epochs"] = 2

    h.train()

    results_dir = find_most_recent_results_dir(h.config, "train")
    best_checkpoints = list(results_dir.glob("*validator_loss=*.pt"))
    assert len(best_checkpoints) == 1, (
        f"Expected 1 validation-loss best-checkpoint file, found: {best_checkpoints}"
    )


def test_best_checkpoint_without_validation(tmp_path):
    """
    When no validator is present, the best checkpoint should fall back to
    training loss scoring.  Verify that a best-checkpoint file is still
    written to the results directory.
    """
    import hyrax

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 2
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    # Only train + infer — no validate group
    h.config["data_request"] = {
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
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    h.train()

    results_dir = find_most_recent_results_dir(h.config, "train")
    best_checkpoints = list(results_dir.glob("*trainer_loss=*.pt"))
    assert len(best_checkpoints) == 1, (
        f"Expected 1 training-loss best-checkpoint file, found: {best_checkpoints}"
    )


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
    h.config["train"]["epochs"] = 2  # run for one more epoch after resuming
    h.config["train"]["resume"] = str(checkpoint_path)

    # Resume training
    h.train()


def test_train_legacy_split_keys_raise_error(tmp_path):
    """Setting legacy split keys in [data_set] raises a RuntimeError
    directing the user to define split fractions in [split]."""
    import hyrax
    from hyrax.pytorch_ignite import setup_dataset

    h = hyrax.Hyrax()
    h.config["data_set"]["train_size"] = 0.6

    with pytest.raises(RuntimeError, match="train_size"):
        setup_dataset(h.config)


def test_train_with_split_defintion(tmp_path):
    """
    Test training with config['split'] fractions on groups sharing the same data_location.
    The train verb calls create_splits which assigns split_indices to each DataProvider
    and writes persisted split files to the results dir.
    """
    import hyrax

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 4
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    shared_location = str(tmp_path / "shared_data")
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": shared_location,
                "primary_id_field": "object_id",
            }
        },
        "validate": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": shared_location,
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

    h.config["split"] = {"train": 0.7, "validate": 0.3, "infer": 1.0, "rng_seed": 42}

    h.config["data_set"]["HyraxRandomDataset"]["size"] = 30
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 42
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    model = h.train()
    assert model is not None

    # Verify split files were persisted inside the train results dir.
    import glob

    npz_files = glob.glob(str(tmp_path / "*-train-*" / "train_split.npz"))
    assert npz_files, "train_split.npz should be persisted in the train results dir"

    # Verify correct sizes via create_splits on a fresh set of providers.
    from hyrax.pytorch_ignite import setup_dataset
    from hyrax.splitting_utils import create_splits

    dataset = setup_dataset(h.config, splits=("train", "validate"))
    result = create_splits(h.config, dataset)

    assert len(result["train"]["indexes"]) == 21  # 70% of 30
    assert len(result["validate"]["indexes"]) == 9  # 30% of 30

    train_set = set(result["train"]["indexes"].tolist())
    validate_set = set(result["validate"]["indexes"].tolist())
    assert train_set.isdisjoint(validate_set)
    assert train_set | validate_set == set(range(30))


def test_train_with_split_definition_dataloader_indices_are_disjoint(tmp_path):
    """
    Verify that create_splits assigns non-overlapping split_indices to the
    train and validate DataProviders when config['split'] is configured.
    """
    import hyrax
    from hyrax.pytorch_ignite import setup_dataset
    from hyrax.splitting_utils import create_splits

    h = hyrax.Hyrax()
    h.config["general"]["results_dir"] = str(tmp_path)

    shared_location = str(tmp_path / "shared_data")
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": shared_location,
                "primary_id_field": "object_id",
            }
        },
        "validate": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": shared_location,
                "primary_id_field": "object_id",
            }
        },
    }
    h.config["split"] = {"train": 0.7, "validate": 0.3, "rng_seed": 42}
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 30
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 42
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    dataset = setup_dataset(h.config, splits=("train", "validate"))
    result = create_splits(h.config, dataset)

    assert len(result["train"]["indexes"]) == 21  # 70% of 30
    assert len(result["validate"]["indexes"]) == 9  # 30% of 30
    assert set(result["train"]["indexes"].tolist()).isdisjoint(set(result["validate"]["indexes"].tolist()))


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


def test_training_info_returned_on_model(loopback_hyrax):
    """
    Test that scheduler works correctly when model is wrapped in DistributedDataParallel.
    This test validates the fix for PR #652 AttributeError bug.
    This test was updated as part of PR #996. It validates the overwriting
    of __getattr__ for DistributedDataParallel which was done in PR #974.
    """
    from unittest.mock import patch

    from torch.nn.parallel import DistributedDataParallel

    h, _ = loopback_hyrax
    gamma = 0.5
    h.config["scheduler"]["name"] = "torch.optim.lr_scheduler.ExponentialLR"
    h.config["torch.optim.lr_scheduler.ExponentialLR"] = {"gamma": gamma}
    h.config["train"]["epochs"] = 2
    initial_lr = 64
    h.config[h.config["optimizer"]["name"]]["lr"] = initial_lr
    h.config["general"]["distributed"] = True

    # Mock idist.auto_model to wrap the model in DistributedDataParallel
    # This simulates what happens in distributed training environments

    def mock_auto_model(model):
        # Wrap the model in DistributedDataParallel to test the fix
        return DistributedDataParallel(model)

    # Patch _auto_model in the pytorch_ignite module
    with patch("hyrax.pytorch_ignite._auto_model", side_effect=mock_auto_model):
        # This should not raise AttributeError: 'DistributedDataParallel' object has no attribute 'scheduler'
        model = h.train()

    # Verify the scheduler worked correctly
    assert hasattr(model, "_learning_rates_history")
    assert hasattr(model, "final_training_metrics")
    assert hasattr(model, "final_validation_metrics")
    expected_history = [[initial_lr * gamma**i] for i in range(h.config["train"]["epochs"])]
    assert model._learning_rates_history == expected_history


def test_train_raises_on_resume_and_model_weights_file(loopback_hyrax, tmp_path):
    """
    Ensure that setting both `resume` and `model_weights_file` in the [train] config
    raises a ValueError before any expensive setup is performed.
    """
    h, _ = loopback_hyrax
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["train"]["resume"] = "/some/checkpoint.pt"
    h.config["train"]["model_weights_file"] = "/some/weights.pth"

    with pytest.raises(ValueError, match="Cannot set both"):
        h.train()


def test_train_with_pretrained_weights(loopback_hyrax, tmp_path):
    """
    Ensure that training can start from pre-trained weights specified via
    config["train"]["model_weights_file"].
    """
    h, _ = loopback_hyrax
    h.config["general"]["results_dir"] = str(tmp_path)

    # First training run to produce a weights file
    h.train()

    results_dir = find_most_recent_results_dir(h.config, "train")
    weights_path = results_dir / h.config["train"]["weights_filename"]
    assert weights_path.exists(), "Expected weights file to exist after first training run"

    # Second training run starting from the pre-trained weights
    h.config["train"]["model_weights_file"] = str(weights_path)
    model = h.train()

    # Verify the config was updated to record the weights file that was used
    assert h.config["train"]["model_weights_file"] == str(weights_path)
    assert model is not None


def test_train_default_model_weights_file_is_false(loopback_hyrax):
    """
    Verify that config["train"]["model_weights_file"] defaults to False and
    that training proceeds normally without loading any pre-existing weights.
    """
    h, _ = loopback_hyrax

    assert not h.config["train"]["model_weights_file"]

    # Training should succeed without any weights file
    model = h.train()
    assert model is not None
