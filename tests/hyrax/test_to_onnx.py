import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_to_onnx_successful_export(tmp_path):
    """Test successful ONNX export from a trained model"""
    import hyrax

    # Create a Hyrax instance with loopback model configuration
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 4
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    # Configure dataset
    h.config["model_inputs"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data_train"),
                "primary_id_field": "object_id",
            }
        },
    }
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    # Train the model
    h.train()

    # Find the training results directory
    from hyrax.config_utils import find_most_recent_results_dir

    train_dir = find_most_recent_results_dir(h.config, "train")
    assert train_dir is not None, "Training results directory should exist"

    # Export to ONNX using the verb
    from hyrax.verbs.to_onnx import ToOnnx

    to_onnx_verb = ToOnnx(h.config)
    to_onnx_verb.run(str(train_dir))

    # Verify ONNX model was created with timestamp-based filename
    onnx_files = list(train_dir.glob("*.onnx"))
    assert len(onnx_files) == 1, "Exactly one ONNX file should be created"

    onnx_file = onnx_files[0]
    # Check filename pattern: <model_name>_opset_<version>_ts_<timestamp>.onnx
    assert "_opset_" in onnx_file.name
    assert "_ts_" in onnx_file.name
    assert onnx_file.suffix == ".onnx"


def test_to_onnx_missing_input_directory(tmp_path):
    """Test handling of missing input directories"""
    import hyrax

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)

    from hyrax.verbs.to_onnx import ToOnnx

    to_onnx_verb = ToOnnx(h.config)

    # Test with non-existent directory
    non_existent_dir = tmp_path / "does_not_exist"
    to_onnx_verb.run(str(non_existent_dir))

    # The verb should log an error and return without creating ONNX files
    # Verify no ONNX files were created
    onnx_files = list(tmp_path.glob("**/*.onnx"))
    assert len(onnx_files) == 0, "No ONNX files should be created for missing directory"


def test_to_onnx_missing_input_directory_from_config(tmp_path):
    """Test handling of missing input directories specified in config"""
    import hyrax

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["onnx"]["input_model_directory"] = str(tmp_path / "does_not_exist")

    from hyrax.verbs.to_onnx import ToOnnx

    to_onnx_verb = ToOnnx(h.config)

    # Test with directory from config that doesn't exist
    to_onnx_verb.run()

    # The verb should log an error and return without creating ONNX files
    onnx_files = list(tmp_path.glob("**/*.onnx"))
    assert len(onnx_files) == 0, "No ONNX files should be created for missing directory"


@pytest.mark.slow
def test_to_onnx_auto_detect_recent_training(tmp_path):
    """Test proper resolution of the most recent training directory"""
    import hyrax

    # Create a Hyrax instance and train a model
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 4
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    # Configure dataset
    h.config["model_inputs"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data_train"),
                "primary_id_field": "object_id",
            }
        },
    }
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    # Train the model
    h.train()

    # Find the training results directory
    from hyrax.config_utils import find_most_recent_results_dir

    train_dir = find_most_recent_results_dir(h.config, "train")
    assert train_dir is not None, "Training results directory should exist"

    # Export to ONNX without specifying input directory (should auto-detect)
    from hyrax.verbs.to_onnx import ToOnnx

    to_onnx_verb = ToOnnx(h.config)
    to_onnx_verb.run()  # No input_model_directory specified

    # Verify ONNX model was created in the training directory
    onnx_files = list(train_dir.glob("*.onnx"))
    assert len(onnx_files) == 1, "ONNX file should be auto-detected and created"


def test_to_onnx_no_previous_training(tmp_path):
    """Test handling when no previous training results exist"""
    import hyrax

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)

    from hyrax.verbs.to_onnx import ToOnnx

    to_onnx_verb = ToOnnx(h.config)

    # Try to export without any prior training
    to_onnx_verb.run()

    # The verb should log an error and return without creating ONNX files
    onnx_files = list(tmp_path.glob("**/*.onnx"))
    assert len(onnx_files) == 0, "No ONNX files should be created without prior training"


@pytest.mark.slow
def test_to_onnx_timestamp_filename_format(tmp_path):
    """Test that the exported ONNX model has the expected timestamp-based filename"""
    import re

    import hyrax

    # Create a Hyrax instance and train a model
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 4
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    # Configure dataset
    h.config["model_inputs"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data_train"),
                "primary_id_field": "object_id",
            }
        },
    }
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    # Train the model
    h.train()

    # Find the training results directory
    from hyrax.config_utils import find_most_recent_results_dir

    train_dir = find_most_recent_results_dir(h.config, "train")

    # Export to ONNX
    from hyrax.verbs.to_onnx import ToOnnx

    to_onnx_verb = ToOnnx(h.config)
    to_onnx_verb.run(str(train_dir))

    # Verify ONNX model filename format
    onnx_files = list(train_dir.glob("*.onnx"))
    assert len(onnx_files) == 1

    onnx_filename = onnx_files[0].name
    # Expected pattern: <model_name>_opset_<version>_ts_<timestamp>.onnx
    # Timestamp format: YYYYMMDD-HHMMSS
    pattern = r"^.+_opset_\d+_ts_\d{8}-\d{6}\.onnx$"
    assert re.match(pattern, onnx_filename), f"ONNX filename '{onnx_filename}' doesn't match expected pattern"


def test_to_onnx_cli_argument_parsing(tmp_path):
    """Test that CLI arguments are properly parsed"""
    import hyrax

    h = hyrax.Hyrax()
    h.config["general"]["results_dir"] = str(tmp_path)

    from hyrax.verbs.to_onnx import ToOnnx

    to_onnx_verb = ToOnnx(h.config)

    # Mock the args object
    class MockArgs:
        def __init__(self):
            self.input_model_directory = str(tmp_path / "test_dir")

    args = MockArgs()

    # This should use the input_model_directory from args
    # We expect it to fail because the directory doesn't exist
    to_onnx_verb.run_cli(args)

    # Verify no ONNX files were created (directory doesn't exist)
    onnx_files = list(tmp_path.glob("**/*.onnx"))
    assert len(onnx_files) == 0
