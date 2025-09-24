from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


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

    if dataset.is_iterable():
        dataset = list(dataset)
        original_dataset_ids = np.array([str(s["object_id"]) for s in dataset])

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
        assert np.all(np.isclose(dataset[dataset_idx]["data"]["image"], inference_results[idx]))


def test_load_model_weights_updates_config_when_auto_detected(tmp_path):
    """Test that config is updated when model_weights_file is auto-detected from train directory"""
    from hyrax.config_utils import ConfigDict
    from hyrax.verbs.infer import Infer

    # Create a mock config with no model_weights_file specified
    config = ConfigDict()
    config["infer"] = {"model_weights_file": None}
    config["train"] = {"weights_filename": "model_weights.pth"}
    config["general"] = {"results_dir": str(tmp_path)}

    # Create a fake train results directory
    train_dir = tmp_path / "20240101-120000-train-abcd"
    train_dir.mkdir(parents=True)
    weights_file = train_dir / "model_weights.pth"
    weights_file.write_text("fake weights content")

    # Create a mock model
    mock_model = MagicMock()

    # Mock find_most_recent_results_dir to return our fake train directory
    with patch("hyrax.config_utils.find_most_recent_results_dir", return_value=train_dir):
        # Call load_model_weights
        Infer.load_model_weights(config, mock_model)

    # Verify that config was updated with the actual weights file path
    assert config["infer"]["model_weights_file"] == str(weights_file)
    # Verify that model.load was called with the correct path
    mock_model.load.assert_called_once_with(weights_file)


def test_load_model_weights_preserves_explicit_config():
    """Test that config is still updated when model_weights_file is explicitly provided"""
    from tempfile import NamedTemporaryFile

    from hyrax.config_utils import ConfigDict
    from hyrax.verbs.infer import Infer

    # Create a temporary weights file
    with NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        tmp_file.write(b"fake weights content")
        weights_path = Path(tmp_file.name)

    try:
        # Create a mock config with explicit model_weights_file
        config = ConfigDict()
        config["infer"] = {"model_weights_file": str(weights_path)}
        config["train"] = {"weights_filename": "model_weights.pth"}

        # Create a mock model
        mock_model = MagicMock()

        # Call load_model_weights
        Infer.load_model_weights(config, mock_model)

        # Verify that config still contains the weights file path (converted to string)
        assert config["infer"]["model_weights_file"] == str(weights_path)
        # Verify that model.load was called with the correct path
        mock_model.load.assert_called_once_with(weights_path)
    finally:
        # Clean up
        weights_path.unlink()
