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
    # trace=5: the trace shim shrinks DataProvider length to 5.  At that size
    # the 20% validation split still yields at least 1 sample
    # (round(5 × 0.2) = 1), which is the minimum needed by the legacy
    # percentage-based split path.  Smaller values (e.g. trace=2) produce an
    # empty validation split and raise a KeyError.
    trace_result = h.train(trace=5)

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
    h.config["train"]["epochs"] = 2  # run for one more epoch after resuming
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


def test_training_info_returned_on_model(loopback_hyrax):
    """
    Test that scheduler works correctly when model is wrapped in DataParallel.
    This test validates the fix for PR #652 AttributeError bug.
    """
    from unittest.mock import patch

    from torch.nn.parallel import DataParallel

    h, _ = loopback_hyrax
    gamma = 0.5
    h.config["scheduler"]["name"] = "torch.optim.lr_scheduler.ExponentialLR"
    h.config["torch.optim.lr_scheduler.ExponentialLR"] = {"gamma": gamma}
    h.config["train"]["epochs"] = 2
    initial_lr = 64
    h.config[h.config["optimizer"]["name"]]["lr"] = initial_lr

    # Mock idist.auto_model to wrap the model in DataParallel
    # This simulates what happens in distributed training environments

    def mock_auto_model(model):
        # Wrap the model in DataParallel to test the fix
        return DataParallel(model)

    # Patch idist.auto_model in the pytorch_ignite module
    with patch("hyrax.pytorch_ignite.idist.auto_model", side_effect=mock_auto_model):
        # This should not raise AttributeError: 'DataParallel' object has no attribute 'scheduler'
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
