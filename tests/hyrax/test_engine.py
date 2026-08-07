"""Unit tests for the Engine verb (src/hyrax/verbs/engine.py).

The Engine verb drives inference against an ONNX-exported model. Several of the
tests below build a real ONNX model by training ``HyraxLoopback`` (an identity
model -- ``forward(x) -> x``) on ``HyraxRandomDataset`` and exporting it with the
``ToOnnx`` verb, then run ``Engine.run()`` against it end-to-end. Because
HyraxLoopback is the identity function, inference results should exactly match
the input images, which makes correctness easy to assert.

Other tests exercise ``Engine``'s helper methods and error-handling paths
directly with lightweight mocks, without paying for a real train/export cycle.
"""

from unittest.mock import MagicMock

import numpy as np

import hyrax
from hyrax.config_utils import find_most_recent_results_dir
from hyrax.datasets.random.hyrax_random_dataset import HyraxRandomDataset
from hyrax.datasets.result_dataset import ResultDataset
from hyrax.verbs.engine import Engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_and_export_onnx(tmp_path, *, split=None, size=20, batch_size=4, seed=24601, shape=(2, 3)):
    """Train HyraxLoopback on HyraxRandomDataset and export it to ONNX.

    Returns (hyrax_instance, onnx_dir, data_location) so callers can both
    reconstruct the ground-truth images and point Engine at the exported model.
    """
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = batch_size
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    data_location = str(tmp_path / "data")
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": data_location,
                "fields": ["image", "label"],
                "primary_id_field": "object_id",
            }
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": data_location,
                "fields": ["image"],
                "primary_id_field": "object_id",
            }
        },
    }
    h.config["split"] = split if split is not None else {"train": 1.0}
    h.config["data_set"]["HyraxRandomDataset"]["size"] = size
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = seed
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = list(shape)

    h.train()
    h.to_onnx()

    onnx_dir = find_most_recent_results_dir(h.config, "onnx")
    assert onnx_dir is not None, "to_onnx should have produced an onnx results directory"

    return h, onnx_dir, data_location


def _reference_images(config, data_location):
    """Rebuild the same HyraxRandomDataset (same seed/size/shape) to get ground truth images.

    Because the infer group's config in `_train_and_export_onnx` doesn't shuffle
    (`_compute_splits` walks the infer group's indices in order, unshuffled) the
    n-th row written to the engine's results always corresponds to dataset index n.
    """
    ds = HyraxRandomDataset(config, data_location)
    return np.stack([ds.get_image(i) for i in range(len(ds))])


def _load_engine_results(config, results_dir):
    return ResultDataset(config, results_dir)


# ---------------------------------------------------------------------------
# End-to-end: correct inference results via a real ONNX-exported model
# ---------------------------------------------------------------------------


def test_engine_run_produces_identity_inference_results(tmp_path):
    """Engine.run() against an onnx-ified HyraxLoopback reproduces the input images."""
    h, onnx_dir, data_location = _train_and_export_onnx(tmp_path)

    verb = Engine(h.config)
    verb.run(model_directory=str(onnx_dir))

    engine_dir = find_most_recent_results_dir(h.config, "engine")
    assert engine_dir is not None

    results = _load_engine_results(h.config, engine_dir)
    expected = _reference_images(h.config, data_location)

    assert len(results) == len(expected)
    for i in range(len(expected)):
        np.testing.assert_allclose(results[i], expected[i], atol=1e-5)


def test_engine_run_writes_correct_object_ids(tmp_path):
    """Engine.run() persists the same object_ids the infer dataset provides."""
    h, onnx_dir, data_location = _train_and_export_onnx(tmp_path)

    verb = Engine(h.config)
    verb.run(model_directory=str(onnx_dir))

    engine_dir = find_most_recent_results_dir(h.config, "engine")
    results = _load_engine_results(h.config, engine_dir)

    reference_ds = HyraxRandomDataset(h.config, data_location)
    expected_ids = [reference_ds.get_object_id(i) for i in range(len(reference_ds))]

    assert results.ids() == expected_ids


def test_engine_run_uses_config_model_directory_when_arg_omitted(tmp_path):
    """When model_directory is not passed to run(), config['engine']['model_directory'] is used."""
    h, onnx_dir, _ = _train_and_export_onnx(tmp_path)
    h.config["engine"]["model_directory"] = str(onnx_dir)

    verb = Engine(h.config)
    verb.run()

    engine_dir = find_most_recent_results_dir(h.config, "engine")
    assert engine_dir is not None
    results = _load_engine_results(h.config, engine_dir)
    assert len(results) == h.config["data_set"]["HyraxRandomDataset"]["size"]


def test_engine_run_finds_most_recent_onnx_dir_automatically(tmp_path):
    """With no explicit arg and no config value, run() auto-discovers the latest onnx export."""
    h, onnx_dir, _ = _train_and_export_onnx(tmp_path)
    assert h.config["engine"]["model_directory"] is False

    verb = Engine(h.config)
    verb.run()

    engine_dir = find_most_recent_results_dir(h.config, "engine")
    assert engine_dir is not None
    results = _load_engine_results(h.config, engine_dir)
    assert len(results) == h.config["data_set"]["HyraxRandomDataset"]["size"]


def test_engine_run_respects_infer_split_fraction(tmp_path):
    """A partial infer split fraction reduces the number of processed/persisted rows."""
    h, onnx_dir, data_location = _train_and_export_onnx(tmp_path, size=20)

    # Only run inference over half the infer dataset.
    h.config["split"] = {"infer": 0.5}

    verb = Engine(h.config)
    verb.run(model_directory=str(onnx_dir))

    engine_dir = find_most_recent_results_dir(h.config, "engine")
    results = _load_engine_results(h.config, engine_dir)
    expected = _reference_images(h.config, data_location)

    assert len(results) == 10
    for i in range(10):
        np.testing.assert_allclose(results[i], expected[i], atol=1e-5)


def test_run_cli_uses_model_directory_from_args(tmp_path):
    """run_cli() forwards args.model_directory into run()."""
    h, onnx_dir, _ = _train_and_export_onnx(tmp_path)

    class MockArgs:
        def __init__(self):
            self.model_directory = str(onnx_dir)

    verb = Engine(h.config)
    verb.run_cli(MockArgs())

    engine_dir = find_most_recent_results_dir(h.config, "engine")
    assert engine_dir is not None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_run_explicit_model_directory_missing_logs_error_and_returns(tmp_path, caplog):
    """A nonexistent explicit model_directory logs an error and returns without running inference."""
    h = hyrax.Hyrax()
    h.config["general"]["results_dir"] = str(tmp_path)
    verb = Engine(h.config)

    missing_dir = tmp_path / "does_not_exist"
    with caplog.at_level("ERROR"):
        result = verb.run(model_directory=str(missing_dir))

    assert result is None
    assert "does not exist" in caplog.text
    assert not any(tmp_path.glob("*-engine-*"))


def test_run_config_model_directory_missing_logs_error_and_returns(tmp_path, caplog):
    """A nonexistent config['engine']['model_directory'] logs an error and returns."""
    h = hyrax.Hyrax()
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["engine"]["model_directory"] = str(tmp_path / "does_not_exist")
    verb = Engine(h.config)

    with caplog.at_level("ERROR"):
        result = verb.run()

    assert result is None
    assert "does not exist" in caplog.text
    assert not any(tmp_path.glob("*-engine-*"))


def test_run_no_onnx_results_found_logs_error_and_returns(tmp_path, caplog):
    """With no explicit arg, no config value, and no prior onnx export, run() logs and returns."""
    h = hyrax.Hyrax()
    h.config["general"]["results_dir"] = str(tmp_path)
    verb = Engine(h.config)

    with caplog.at_level("ERROR"):
        result = verb.run()

    assert result is None
    assert "No previous training results directory" in caplog.text
    assert not any(tmp_path.glob("*-engine-*"))


def test_run_missing_prepare_inputs_file_logs_error_and_returns(tmp_path, caplog):
    """If the resolved model directory has no prepare_inputs.py, run() logs and returns
    before creating an engine results directory."""
    h, onnx_dir, _ = _train_and_export_onnx(tmp_path)
    (onnx_dir / "prepare_inputs.py").unlink()

    verb = Engine(h.config)
    with caplog.at_level("ERROR"):
        result = verb.run(model_directory=str(onnx_dir))

    assert result is None
    assert "No prepare_inputs function found" in caplog.text
    assert not any(tmp_path.glob("*-engine-*"))


# ---------------------------------------------------------------------------
# Helper methods, in isolation (no training required)
# ---------------------------------------------------------------------------


def _fake_ort_session(input_names):
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name=n) for n in input_names]
    for mock_input, name in zip(session.get_inputs.return_value, input_names):
        mock_input.name = name
    return session


def test_create_ort_inputs_tuple_skips_empty_elements():
    """create_ort_inputs only includes tuple elements that are non-empty,
    matching them positionally to the ONNX session's declared inputs."""
    verb = Engine.__new__(Engine)
    verb.ort_session = _fake_ort_session(["image_input"])

    image = np.zeros((2, 3), dtype=np.float32)
    label = np.array([], dtype=np.float32)  # empty -> should be skipped

    ort_inputs = verb.create_ort_inputs((image, label))

    assert ort_inputs == {"image_input": image}


def test_create_ort_inputs_multiple_nonempty_elements():
    """Both tuple elements are included, keyed by their respective ONNX input names,
    when neither is empty."""
    verb = Engine.__new__(Engine)
    verb.ort_session = _fake_ort_session(["image_input", "label_input"])

    image = np.zeros((2, 3), dtype=np.float32)
    label = np.ones((1,), dtype=np.float32)

    ort_inputs = verb.create_ort_inputs((image, label))

    assert set(ort_inputs.keys()) == {"image_input", "label_input"}
    np.testing.assert_array_equal(ort_inputs["image_input"], image)
    np.testing.assert_array_equal(ort_inputs["label_input"], label)


def test_create_ort_inputs_non_tuple_input():
    """A non-tuple prepared_batch is keyed directly under the first declared input name."""
    verb = Engine.__new__(Engine)
    verb.ort_session = _fake_ort_session(["only_input"])

    image = np.zeros((2, 3), dtype=np.float32)
    ort_inputs = verb.create_ort_inputs(image)

    assert ort_inputs == {"only_input": image}


def test_run_onnx_batch_delegates_to_session_run():
    """run_onnx_batch calls ort_session.run(None, ort_inputs) and returns its result."""
    verb = Engine.__new__(Engine)
    verb.ort_session = MagicMock()
    verb.ort_session.run.return_value = ["sentinel_result"]

    ort_inputs = {"input": np.zeros((1,))}
    result = verb.run_onnx_batch(ort_inputs)

    verb.ort_session.run.assert_called_once_with(None, ort_inputs)
    assert result == ["sentinel_result"]


def test_setup_trace_returns_same_fn_when_no_trace_active():
    """_setup_trace is a no-op when tracing is not enabled (get_trace() is None)."""
    verb = Engine.__new__(Engine)

    def prepare_inputs_fn(batch):
        return batch

    result = verb._setup_trace(prepare_inputs_fn)

    assert result is prepare_inputs_fn


# ---------------------------------------------------------------------------
# Verb metadata / CLI parser
# ---------------------------------------------------------------------------


def test_verb_metadata():
    """Engine declares the expected cli_name and description."""
    assert Engine.cli_name == "engine"
    assert "ONNX" in Engine.description


def test_setup_parser_adds_model_directory_argument():
    """setup_parser() registers --model-directory as an optional string argument."""
    import argparse

    parser = argparse.ArgumentParser()
    Engine.setup_parser(parser)

    args = parser.parse_args([])
    assert args.model_directory is None

    args = parser.parse_args(["--model-directory", "/some/path"])
    assert args.model_directory == "/some/path"
