from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from ignite.engine import Engine

from hyrax.pytorch_ignite import (
    HyraxEvents,
    _attach_epoch_metric_averaging,
    _batch_sample_count,
    _inner_loop,
    create_tester,
    create_trainer,
    create_validator,
    results_to_numpy,
    setup_dataset,
)

# Two batches of unequal size: 2 samples with mean loss 0, then 1 sample with mean loss 4.
# Sample-weighted epoch mean is (0 * 2 + 4 * 1) / 3 = 4/3; an unweighted batch mean would give 2.
UNEQUAL_BATCHES = [
    {"object_id": ["a", "b"], "loss": 0.0},
    {"object_id": ["c"], "loss": 4.0},
]


class TestSetupDataset:
    """Tests for the setup_dataset function."""

    def test_setup_dataset_missing_dataset_class_raises_error(self):
        """Test that missing dataset_class raises appropriate RuntimeError."""
        # Create a minimal config that omits dataset_class in data_request
        config = {
            "data_set": {"seed": False},
            "data_request": {
                "train": {
                    "test_dataset": {
                        # Intentionally missing "dataset_class"
                        "data_location": "/some/path"
                    },
                },
                "infer": {
                    "test_dataset": {
                        # Intentionally missing "dataset_class"
                        "data_location": "/some/path"
                    },
                },
            },
        }

        # This should raise RuntimeError from DataProvider when dataset_class is missing
        with pytest.raises(RuntimeError) as exc_info:
            setup_dataset(config)

        assert "does not specify a 'dataset_class'" in str(exc_info.value)

    def test_setup_dataset_invalid_dataset_class_raises_error(self):
        """Test that providing an invalid dataset_class raises appropriate ValueError."""
        # Create a config with an invalid dataset_class
        config = {
            "data_set": {"seed": False},
            "data_request": {
                "train": {
                    "test_dataset": {
                        "dataset_class": "NonExistentDatasetClass",
                        "data_location": "/some/path",
                    }
                },
                "infer": {
                    "test_dataset": {
                        "dataset_class": "NonExistentDatasetClass",
                        "data_location": "/some/path",
                    },
                },
            },
        }

        # DataProvider will try to look up the class in the registry and fail
        with pytest.raises(ValueError) as exc_info:
            setup_dataset(config)

        assert "Class name NonExistentDatasetClass" in str(exc_info.value)

    def test_setup_dataset_missing_data_location_uses_none(self):
        """Test that setup_dataset creates DataProvider instances even when data_location is absent."""
        # Create a config with dataset_class but no data_location
        config = {
            "data_request": {
                "train": {
                    "test_dataset": {
                        "dataset_class": "HyraxRandomDataset"
                        # Intentionally missing "data_location"
                    },
                },
                "infer": {
                    "test_dataset": {
                        "dataset_class": "HyraxRandomDataset"
                        # Intentionally missing "data_location"
                    },
                },
            }
        }

        # Use a real class (not MagicMock) so isinstance() in setup_dataset doesn't TypeError.
        # __new__ intercepts construction and returns our sentinel instance.
        sentinel = MagicMock()
        provider_calls = []

        class MockDataProvider:
            def __new__(cls, config, request):
                provider_calls.append((config, request))
                return sentinel

        with patch("hyrax.pytorch_ignite.DataProvider", MockDataProvider):
            result = setup_dataset(config)

            # DataProvider should be created for each data group
            assert len(provider_calls) == 2
            assert (config, config["data_request"]["train"]) in provider_calls
            assert (config, config["data_request"]["infer"]) in provider_calls
            assert result["train"] is sentinel
            assert result["infer"] is sentinel

    def test_setup_dataset_with_both_keys_present(self):
        """Test normal case where both dataset_class and data_location are present."""
        # Create a complete config
        config = {
            "data_request": {
                "train": {
                    "test_dataset": {
                        "dataset_class": "HyraxRandomDataset",
                        "data_location": "/some/valid/path",
                    },
                },
                "infer": {
                    "test_dataset": {
                        "dataset_class": "HyraxRandomDataset",
                        "data_location": "/some/valid/path",
                    },
                },
            }
        }

        # Use a real class (not MagicMock) so isinstance() in setup_dataset doesn't TypeError.
        sentinel = MagicMock()
        provider_calls = []

        class MockDataProvider:
            def __new__(cls, config, request):
                provider_calls.append((config, request))
                return sentinel

        with patch("hyrax.pytorch_ignite.DataProvider", MockDataProvider):
            result = setup_dataset(config)

            # DataProvider should be created for each data group with the full request dict
            assert len(provider_calls) == 2
            assert (config, config["data_request"]["train"]) in provider_calls
            assert (config, config["data_request"]["infer"]) in provider_calls
            assert result["train"] is sentinel
            assert result["infer"] is sentinel


class TestInnerLoop:
    """Tests for the _inner_loop function, specifically focused on None handling."""

    def test_inner_loop_with_tuple_containing_none(self):
        """Test that _inner_loop handles tuples with None values correctly."""
        # Create test data
        test_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        test_batch = (test_array, None)  # Simulate missing labels

        # Mock functions
        def mock_prepare_inputs(batch):
            return batch

        mock_func = MagicMock(return_value={"loss": 1.0})
        device = torch.device("cpu")
        config = {}
        engine = MagicMock()

        # Call _inner_loop
        _inner_loop(mock_func, mock_prepare_inputs, device, config, engine, test_batch)

        # Verify the function was called with a tuple where the second element is None
        assert mock_func.called
        called_batch = mock_func.call_args[0][0]
        assert isinstance(called_batch, tuple)
        assert len(called_batch) == 2
        assert isinstance(called_batch[0], torch.Tensor)
        assert called_batch[1] is None

    def test_inner_loop_with_none_batch(self):
        """Test that _inner_loop handles None batch correctly."""
        # Create test data
        test_batch = None

        # Mock functions
        def mock_prepare_inputs(batch):
            return batch

        mock_func = MagicMock(return_value={"loss": 1.0})
        device = torch.device("cpu")
        config = {}
        engine = MagicMock()

        # Call _inner_loop - should not raise an error
        _inner_loop(mock_func, mock_prepare_inputs, device, config, engine, test_batch)

        # Verify the function was called with None
        assert mock_func.called
        called_batch = mock_func.call_args[0][0]
        assert called_batch is None

    def test_inner_loop_with_normal_tuple(self):
        """Test that _inner_loop still works correctly with normal tuples (no None)."""
        # Create test data
        test_array1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        test_array2 = np.array([0, 1], dtype=np.int64)
        test_batch = (test_array1, test_array2)

        # Mock functions
        def mock_prepare_inputs(batch):
            return batch

        mock_func = MagicMock(return_value={"loss": 1.0})
        device = torch.device("cpu")
        config = {}
        engine = MagicMock()

        # Call _inner_loop
        _inner_loop(mock_func, mock_prepare_inputs, device, config, engine, test_batch)

        # Verify the function was called with tensors
        assert mock_func.called
        called_batch = mock_func.call_args[0][0]
        assert isinstance(called_batch, tuple)
        assert len(called_batch) == 2
        assert isinstance(called_batch[0], torch.Tensor)
        assert isinstance(called_batch[1], torch.Tensor)

    def test_inner_loop_with_single_array(self):
        """Test that _inner_loop works with a single array (not a tuple)."""
        # Create test data
        test_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        # Mock functions
        def mock_prepare_inputs(batch):
            return batch

        mock_func = MagicMock(return_value={"loss": 1.0})
        device = torch.device("cpu")
        config = {}
        engine = MagicMock()

        # Call _inner_loop
        _inner_loop(mock_func, mock_prepare_inputs, device, config, engine, test_array)

        # Verify the function was called with a tensor
        assert mock_func.called
        called_batch = mock_func.call_args[0][0]
        assert isinstance(called_batch, torch.Tensor)


class TestResultsToNumpy:
    """Tests for the results_to_numpy function."""

    def test_tensor_becomes_array(self):
        """A single tensor with autograd history becomes a detached numpy array."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True) * 2

        result = results_to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[2.0, 4.0], [6.0, 8.0]])

    def test_dict_becomes_dict_of_arrays(self):
        """A dict of tensors becomes a dict with the same keys and numpy array values."""
        results = {
            "a": torch.tensor([1.0, 2.0], requires_grad=True) * 2,
            "b": torch.ones((2, 2, 2)),
        }

        converted = results_to_numpy(results)

        assert set(converted) == {"a", "b"}
        assert all(isinstance(value, np.ndarray) for value in converted.values())
        np.testing.assert_array_equal(converted["a"], [2.0, 4.0])
        assert converted["b"].shape == (2, 2, 2)

    def test_other_type_raises(self):
        """Anything other than a tensor or dict of tensors raises a RuntimeError."""
        with pytest.raises(RuntimeError, match="Expected batch_results"):
            results_to_numpy([torch.ones(2)])


def _record_epoch_outputs(engine):
    """Collect engine.state.output at every HYRAX_EPOCH_COMPLETED, after averaging has run."""
    outputs = []
    engine.add_event_handler(HyraxEvents.HYRAX_EPOCH_COMPLETED, lambda e: outputs.append(e.state.output))
    return outputs


class _UnequalBatchModel(torch.nn.Module):
    """Stub model whose per-batch loss is read straight from the batch."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        self.scheduler = None

    @staticmethod
    def prepare_inputs(batch):
        return np.array(batch["loss"], dtype=np.float32)

    def train_batch(self, loss):
        return {"loss": loss}

    def validate_batch(self, loss):
        return {"loss": loss}

    def test_batch(self, loss):
        return {"loss": loss}


class TestEpochMetricAveraging:
    """Epoch metrics must be averaged over samples, not over batches."""

    def test_batch_sample_count_uses_object_id(self):
        """The sample count comes from the object_id list of a collated batch."""
        assert _batch_sample_count({"object_id": ["a", "b", "c"]}) == 3

    def test_batch_sample_count_falls_back_to_one(self):
        """Batches without object_id are weighted equally."""
        assert _batch_sample_count({"data": [1, 2]}) == 1
        assert _batch_sample_count(torch.ones(4)) == 1

    def test_unequal_batches_are_sample_weighted(self):
        """Batches of 2 and 1 samples with mean losses 0 and 4 average to 4/3."""
        engine = Engine(lambda e, batch: {"loss": torch.tensor(batch["loss"])})
        _attach_epoch_metric_averaging(engine)
        outputs = _record_epoch_outputs(engine)

        engine.run(UNEQUAL_BATCHES, max_epochs=1)

        assert outputs[0]["loss"] == pytest.approx(4 / 3)
        assert isinstance(outputs[0]["loss"], float)

    def test_sums_reset_each_epoch(self):
        """Each epoch is averaged independently of the previous one."""
        engine = Engine(lambda e, batch: {"loss": batch["loss"] * e.state.epoch})
        _attach_epoch_metric_averaging(engine)
        outputs = _record_epoch_outputs(engine)

        engine.run(UNEQUAL_BATCHES, max_epochs=2)

        assert outputs[0]["loss"] == pytest.approx(4 / 3)
        # Without a reset this would be (0 + 4 + 0 + 8) / 6 = 2.
        assert outputs[1]["loss"] == pytest.approx(8 / 3)

    @patch("hyrax.pytorch_ignite.mlflow")
    def test_validator_unequal_batches(self, _mlflow):
        """create_validator reports the sample-weighted validation loss."""
        model = _UnequalBatchModel()
        # create_validator hooks the trainer's HYRAX_EPOCH_COMPLETED, so the stub trainer needs it.
        trainer = Engine(lambda e, b: None)
        trainer.register_events(*HyraxEvents)
        validator = create_validator(model, {}, UNEQUAL_BATCHES, trainer)
        outputs = _record_epoch_outputs(validator)

        validator.run(UNEQUAL_BATCHES)

        assert outputs[0]["loss"] == pytest.approx(4 / 3)
        assert model.final_validation_metrics["loss"] == pytest.approx(4 / 3)

    @patch("hyrax.pytorch_ignite.mlflow")
    def test_tester_unequal_batches(self, _mlflow):
        """create_tester reports the sample-weighted test loss."""
        tester = create_tester(_UnequalBatchModel(), {})
        outputs = _record_epoch_outputs(tester)

        tester.run(UNEQUAL_BATCHES)

        assert outputs[0]["loss"] == pytest.approx(4 / 3)

    @patch("hyrax.pytorch_ignite.mlflow")
    def test_trainer_unequal_batches(self, _mlflow, tmp_path):
        """create_trainer reports the sample-weighted training loss every epoch."""
        model = _UnequalBatchModel()
        trainer = create_trainer(model, {"train": {"resume": False}}, tmp_path)
        outputs = _record_epoch_outputs(trainer)

        trainer.run(UNEQUAL_BATCHES, max_epochs=2)

        assert [o["loss"] for o in outputs] == pytest.approx([4 / 3, 4 / 3])
