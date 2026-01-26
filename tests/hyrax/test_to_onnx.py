import logging

import pytest

import hyrax
from hyrax.config_utils import find_most_recent_results_dir
from hyrax.verbs.to_onnx import ToOnnx

logger = logging.getLogger(__name__)


@pytest.fixture
def trained_hyrax(tmp_path):
    """Fixture that creates a trained Hyrax instance for ONNX export tests"""
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
                "fields": ["image", "label"],
                "primary_id_field": "object_id",
            }
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data_train"),
                "fields": ["image"],
                "primary_id_field": "object_id",
            }
        },
    }
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    # Train the model
    h.train()

    return h


def test_to_onnx_successful_export(trained_hyrax):
    """Test successful ONNX export from a trained model"""
    h = trained_hyrax

    # Find the training results directory
    train_dir = find_most_recent_results_dir(h.config, "train")
    assert train_dir is not None, "Training results directory should exist"

    # Export to ONNX using the verb
    to_onnx_verb = ToOnnx(h.config)
    to_onnx_verb.run(str(train_dir))

    onnx_dir = find_most_recent_results_dir(h.config, "onnx")

    # Verify ONNX model was created with timestamp-based filename
    onnx_files = list(onnx_dir.glob("*.onnx"))
    assert len(onnx_files) == 1, "Exactly one ONNX file should be created"

    onnx_file = onnx_files[0]
    # Check filename pattern: <model_name>_opset_<version>.onnx (opset version only)
    assert "model" in onnx_file.name
    assert onnx_file.suffix == ".onnx"


def test_to_onnx_missing_input_directory(tmp_path):
    """Test handling of missing input directories"""
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)

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
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["onnx"]["input_model_directory"] = str(tmp_path / "does_not_exist")

    to_onnx_verb = ToOnnx(h.config)

    # Test with directory from config that doesn't exist
    to_onnx_verb.run()

    # The verb should log an error and return without creating ONNX files
    onnx_files = list(tmp_path.glob("**/*.onnx"))
    assert len(onnx_files) == 0, "No ONNX files should be created for missing directory"


def test_to_onnx_no_previous_training(tmp_path):
    """Test handling when no previous training results exist"""
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)

    to_onnx_verb = ToOnnx(h.config)

    # Try to export without any prior training
    to_onnx_verb.run()

    # The verb should log an error and return without creating ONNX files
    onnx_files = list(tmp_path.glob("**/*.onnx"))
    assert len(onnx_files) == 0, "No ONNX files should be created without prior training"


def test_to_onnx_cli_argument_parsing(tmp_path):
    """Test that CLI arguments are properly parsed"""
    h = hyrax.Hyrax()
    h.config["general"]["results_dir"] = str(tmp_path)

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
