import onnxruntime

from hyrax import plugin_utils, pytorch_ignite, splitting_utils
from hyrax.datasets import result_factories
from hyrax.verbs.engine import Engine


def test_engine_does_not_call_load_to_tensor(tmp_path, monkeypatch):
    """Engine should rely only on prepare_inputs and never call load_to_tensor."""
    calls = {"load_to_tensor": 0, "commit": 0}

    def _prepare_inputs(_path):
        return lambda batch: batch

    def _load_to_tensor(_path):
        calls["load_to_tensor"] += 1
        raise AssertionError("load_to_tensor should not be called by engine")

    class _MockSession:
        def __init__(self, _onnx_path):
            pass

    class _MockDataset:
        split_indices = None

        def on_epoch_start(self, _verb_name):
            pass

        def __len__(self):
            return 0

        def collate(self, batch):
            return batch

    class _MockResultsWriter:
        def write_batch(self, _object_ids, _results):
            pass

        def commit(self):
            calls["commit"] += 1

    monkeypatch.setattr(plugin_utils, "load_prepare_inputs", _prepare_inputs)
    monkeypatch.setattr(plugin_utils, "load_to_tensor", _load_to_tensor)
    monkeypatch.setattr(onnxruntime, "InferenceSession", _MockSession)
    monkeypatch.setattr(
        pytorch_ignite,
        "setup_dataset",
        lambda *_args, **_kwargs: {"infer": _MockDataset()},
    )
    monkeypatch.setattr(splitting_utils, "create_splits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        result_factories,
        "create_results_writer",
        lambda *_args, **_kwargs: _MockResultsWriter(),
    )

    config = {
        "engine": {"model_directory": str(tmp_path)},
        "data_loader": {"batch_size": 4},
        "general": {"results_dir": str(tmp_path)},
    }
    Engine(config).run()

    assert calls["load_to_tensor"] == 0
    assert calls["commit"] == 1
