from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from hyrax.pytorch_ignite import _inner_loop, setup_dataset


class TestSetupDataset:
    """Tests for the setup_dataset function, specifically focused on iterable dataset handling."""

    def test_setup_dataset_missing_dataset_class_raises_error(self):
        """Test that missing dataset_class raises appropriate RuntimeError."""
        # Create a minimal config that would trigger iterable dataset path
        config = {
            "model_inputs": {
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

        # Mock the functions that would be called before our code
        with patch("hyrax.data_sets.data_provider.generate_data_request_from_config") as mock_generate:
            with patch("hyrax.pytorch_ignite.is_iterable_dataset_requested") as mock_is_iterable:
                # Set up mocks to trigger the iterable dataset path
                mock_generate.return_value = config["model_inputs"]
                mock_is_iterable.return_value = True

                # This should raise RuntimeError with our specific message
                with pytest.raises(RuntimeError) as exc_info:
                    setup_dataset(config)

                assert "dataset_class must be specified in 'data_request'." in str(exc_info.value)

    def test_setup_dataset_invalid_dataset_class_raises_error(self):
        """Test that providing an invalid dataset_class raises appropriate RuntimeError."""
        # Create a config with an invalid dataset_class
        config = {
            "model_inputs": {
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

        # Mock the functions that would be called before our code
        with patch("hyrax.data_sets.data_provider.generate_data_request_from_config") as mock_generate:
            with patch("hyrax.pytorch_ignite.is_iterable_dataset_requested") as mock_is_iterable:
                # Set up mocks to trigger the iterable dataset path
                mock_generate.return_value = config["model_inputs"]
                mock_is_iterable.return_value = True
                # Make the registry lookup fail by simulating the dataset class not being in registry

                # This should raise RuntimeError with our specific message
                with pytest.raises(ValueError) as exc_info:
                    setup_dataset(config)

                assert "Class name NonExistentDatasetClass" in str(exc_info.value)

    def test_setup_dataset_missing_data_location_uses_none(self):
        """Test that missing data_location passes None to dataset constructor."""
        # Create a config with dataset_class but no data_location
        config = {
            "model_inputs": {
                "train": {
                    "test_dataset": {
                        "dataset_class": "HyraxRandomIterableDataset"
                        # Intentionally missing "data_location"
                    },
                },
                "infer": {
                    "test_dataset": {
                        "dataset_class": "HyraxRandomIterableDataset"
                        # Intentionally missing "data_location"
                    },
                },
            }
        }

        # Mock dataset class and registry
        mock_dataset_instance = MagicMock()
        mock_dataset_cls = MagicMock(return_value=mock_dataset_instance)

        with patch("hyrax.data_sets.data_provider.generate_data_request_from_config") as mock_generate:
            with patch("hyrax.pytorch_ignite.is_iterable_dataset_requested") as mock_is_iterable:
                with patch("hyrax.pytorch_ignite.fetch_dataset_class") as mock_fetch_cls:
                    # Set up mocks
                    mock_generate.return_value = config["model_inputs"]
                    mock_is_iterable.return_value = True
                    mock_fetch_cls.return_value = mock_dataset_cls

                    # Call the function
                    result = setup_dataset(config)

                    # Verify the dataset constructor was called with data_location=None
                    expected_call = call(config=config, data_location=None)
                    assert mock_dataset_cls.call_count == 2
                    mock_dataset_cls.assert_has_calls([expected_call, expected_call])
                    assert result["train"] == mock_dataset_instance
                    assert result["infer"] == mock_dataset_instance

    def test_setup_dataset_with_both_keys_present(self):
        """Test normal case where both dataset_class and data_location are present."""
        # Create a complete config
        config = {
            "model_inputs": {
                "train": {
                    "test_dataset": {
                        "dataset_class": "HyraxRandomIterableDataset",
                        "data_location": "/some/valid/path",
                    },
                },
                "infer": {
                    "test_dataset": {
                        "dataset_class": "HyraxRandomIterableDataset",
                        "data_location": "/some/valid/path",
                    },
                },
            }
        }

        # Mock dataset class and registry
        mock_dataset_instance = MagicMock()
        mock_dataset_cls = MagicMock(return_value=mock_dataset_instance)

        with patch("hyrax.data_sets.data_provider.generate_data_request_from_config") as mock_generate:
            with patch("hyrax.pytorch_ignite.is_iterable_dataset_requested") as mock_is_iterable:
                with patch("hyrax.pytorch_ignite.fetch_dataset_class") as mock_fetch_cls:
                    # Set up mocks
                    mock_generate.return_value = config["model_inputs"]
                    mock_is_iterable.return_value = True
                    mock_fetch_cls.return_value = mock_dataset_cls

                    # Call the function
                    result = setup_dataset(config)

                    # Verify the dataset constructor was called with correct parameters
                    expected_call = call(config=config, data_location="/some/valid/path")
                    assert mock_dataset_cls.call_count == 2
                    mock_dataset_cls.assert_has_calls([expected_call, expected_call])
                    assert result["train"] == mock_dataset_instance
                    assert result["infer"] == mock_dataset_instance


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
