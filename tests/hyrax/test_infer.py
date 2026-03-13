from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_infer_trace(loopback_hyrax):
    """
    Integration test: infer(trace=N) returns a TraceResult that can be
    printed and inspected stage by stage, modelling how a user would use
    the trace feature from a notebook.
    """
    from hyrax.trace import TraceResult, TraceStage

    h, _ = loopback_hyrax
    # trace=5: keeps the traced batch small while ensuring the DataProvider is
    # large enough for the percentage-based split path to produce non-empty
    # partitions (round(5 × 0.2) = 1 sample per 20% split).
    trace_result = h.infer(trace=5)

    # User would first print the result
    assert isinstance(trace_result, TraceResult)
    assert len(str(trace_result)) > 0

    # User accesses stages via attribute notation: trace_result.resolve_data
    assert isinstance(trace_result.resolve_data, TraceStage)

    # User accesses stages via dict notation: trace_result["collate"]
    assert isinstance(trace_result["collate"], TraceStage)

    # Stages should have captured calls from the data pipeline
    assert len(trace_result["resolve_data"]) > 0
    assert len(trace_result["collate"]) > 0

    # User can dive into individual calls within a stage
    first_resolve_call = trace_result["resolve_data"][0]
    assert str(first_resolve_call)


@pytest.mark.parametrize("shuffle", [True, False])
def test_infer_order(loopback_hyrax, shuffle):
    """Test that the order of data run through infer
    is correct in the presence of several splits
    """
    h, dataset = loopback_hyrax
    h.config["data_loader"]["shuffle"] = shuffle

    dataset = dataset["infer"]
    inference_results = h.infer()
    inference_result_ids = inference_results.ids()
    original_dataset_ids = dataset.ids()

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
    from hyrax.models.model_utils import load_model_weights

    # Create a mock config with no model_weights_file specified
    config = {}
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
        load_model_weights(config, mock_model, "infer")

    # Verify that config was updated with the actual weights file path
    assert config["infer"]["model_weights_file"] == str(weights_file)
    # Verify that model.load was called with the correct path
    mock_model.load.assert_called_once_with(weights_file)


def test_load_model_weights_does_not_mutate_original_when_copied(tmp_path):
    """Verify that passing a deepcopy to load_model_weights protects the original dict.

    This is the core mechanism behind the issue #703 fix: Hyrax passes a deepcopy
    of its config to verbs, so mutations in load_model_weights stay local.
    """
    from copy import deepcopy

    from hyrax.models.model_utils import load_model_weights

    original_config = {
        "infer": {"model_weights_file": None},
        "train": {"weights_filename": "model_weights.pth"},
        "general": {"results_dir": str(tmp_path)},
    }

    train_dir = tmp_path / "20240101-120000-train-abcd"
    train_dir.mkdir(parents=True)
    weights_file = train_dir / "model_weights.pth"
    weights_file.write_text("fake weights content")

    config_copy = deepcopy(original_config)
    mock_model = MagicMock()

    with patch("hyrax.config_utils.find_most_recent_results_dir", return_value=train_dir):
        load_model_weights(config_copy, mock_model, "infer")

    # The copy should be mutated
    assert config_copy["infer"]["model_weights_file"] == str(weights_file)
    # The original must remain untouched
    assert original_config["infer"]["model_weights_file"] is None


def test_load_model_weights_preserves_explicit_config():
    """Test that config is still updated when model_weights_file is explicitly provided"""
    from tempfile import NamedTemporaryFile

    from hyrax.models.model_utils import load_model_weights

    # Create a temporary weights file
    with NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        tmp_file.write(b"fake weights content")
        weights_path = Path(tmp_file.name)

    try:
        # Create a mock config with explicit model_weights_file
        config = {}
        config["infer"] = {"model_weights_file": str(weights_path)}
        config["train"] = {"weights_filename": "model_weights.pth"}

        # Create a mock model
        mock_model = MagicMock()

        # Call load_model_weights
        load_model_weights(config, mock_model, "infer")

        # Verify that config still contains the weights file path (converted to string)
        assert config["infer"]["model_weights_file"] == str(weights_path)
        # Verify that model.load was called with the correct path
        mock_model.load.assert_called_once_with(weights_path)
    finally:
        # Clean up
        weights_path.unlink()
