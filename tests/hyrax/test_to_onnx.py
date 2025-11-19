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


@pytest.fixture
def trained_hyrax_supervised(tmp_path):
    """Fixture that creates a trained Hyrax instance with supervised data (image + label)
    for ONNX export tests"""
    # Create a Hyrax instance with loopback model configuration
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 4
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    # Configure dataset with both image and label fields
    h.config["model_inputs"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data_train"),
                "fields": ["image", "label"],
                "primary_id_field": "object_id",
            }
        },
    }
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]
    # Use numeric labels (0, 1, 2) instead of strings so they can be converted to tensors
    h.config["data_set"]["HyraxRandomDataset"]["provided_labels"] = [0, 1, 2]

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

    # Verify ONNX model was created with timestamp-based filename
    onnx_files = list(train_dir.glob("*.onnx"))
    assert len(onnx_files) == 1, "Exactly one ONNX file should be created"

    onnx_file = onnx_files[0]
    # Check filename pattern: <model_name>_opset_<version>_ts_<timestamp>.onnx
    assert "_opset_" in onnx_file.name
    assert "_ts_" in onnx_file.name
    assert onnx_file.suffix == ".onnx"


def test_to_onnx_supervised_export(trained_hyrax_supervised):
    """Test ONNX export from a trained supervised model with (image, label) data.

    This test demonstrates the issue where ONNX export fails for supervised models
    that use (data, label) tuples during training. The current implementation
    in model_exporters.py assumes sample is a simple tensor, not a tuple.

    The test currently fails at the export stage with:
    AttributeError: 'tuple' object has no attribute 'numpy'

    This happens because:
    1. During training, to_tensor() returns (image_tensor, label_tensor) for supervised models
    2. The export code passes this tuple to model(sample) which works fine
    3. But then it tries to call sample.numpy() which fails because sample is a tuple

    Even if we fix that, ONNX tracing will prune the label input if it's not used
    in the forward pass, creating a model that only accepts data (not data+label).
    """
    import pytest

    h = trained_hyrax_supervised

    # Find the training results directory
    train_dir = find_most_recent_results_dir(h.config, "train")
    assert train_dir is not None, "Training results directory should exist"

    # Export to ONNX using the verb - this should fail with AttributeError
    to_onnx_verb = ToOnnx(h.config)

    # The export currently fails because model_exporters.py expects sample to be
    # a tensor, but for supervised models it's a tuple (data, label).
    with pytest.raises(AttributeError, match="'tuple' object has no attribute 'numpy'"):
        to_onnx_verb.run(str(train_dir))


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


def test_to_onnx_auto_detect_recent_training(trained_hyrax):
    """Test proper resolution of the most recent training directory"""
    h = trained_hyrax

    # Find the training results directory
    train_dir = find_most_recent_results_dir(h.config, "train")
    assert train_dir is not None, "Training results directory should exist"

    # Export to ONNX without specifying input directory (should auto-detect)
    to_onnx_verb = ToOnnx(h.config)
    to_onnx_verb.run()  # No input_model_directory specified

    # Verify ONNX model was created in the training directory
    onnx_files = list(train_dir.glob("*.onnx"))
    assert len(onnx_files) == 1, "ONNX file should be auto-detected and created"


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
