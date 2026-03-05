from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from hyrax.pytorch_ignite import _inner_loop, setup_dataset


class TestSetupDataset:
    """Tests for the setup_dataset function."""

    def test_setup_dataset_missing_dataset_class_raises_error(self):
        """Test that missing dataset_class raises appropriate RuntimeError."""
        # Create a minimal config that omits dataset_class in data_request
        config = {
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
            }
        }

        # This should raise RuntimeError from DataProvider when dataset_class is missing
        with pytest.raises(RuntimeError) as exc_info:
            setup_dataset(config)

        assert "does not specify a 'dataset_class'" in str(exc_info.value)

    def test_setup_dataset_invalid_dataset_class_raises_error(self):
        """Test that providing an invalid dataset_class raises appropriate ValueError."""
        # Create a config with an invalid dataset_class
        config = {
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
            }
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
        sentinel.split_fraction = None
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
        sentinel.split_fraction = None
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
