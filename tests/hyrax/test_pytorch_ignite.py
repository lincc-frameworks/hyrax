from unittest.mock import MagicMock, patch

import pytest

from hyrax.pytorch_ignite import setup_dataset


class TestSetupDataset:
    """Tests for the setup_dataset function, specifically focused on iterable dataset handling."""

    def test_setup_dataset_missing_dataset_class_raises_error(self):
        """Test that missing dataset_class raises appropriate RuntimeError."""
        # Create a minimal config that would trigger iterable dataset path
        config = {
            "model_inputs": {
                "test_dataset": {
                    # Intentionally missing "dataset_class"
                    "data_location": "/some/path"
                }
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

                assert "dataset_class must be specified in 'model_inputs'." in str(exc_info.value)

    def test_setup_dataset_invalid_dataset_class_raises_error(self):
        """Test that providing an invalid dataset_class raises appropriate RuntimeError."""
        # Create a config with an invalid dataset_class
        config = {
            "model_inputs": {
                "test_dataset": {"dataset_class": "NonExistentDatasetClass", "data_location": "/some/path"}
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
                "test_dataset": {
                    "dataset_class": "HyraxRandomIterableDataset"
                    # Intentionally missing "data_location"
                }
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
                    mock_dataset_cls.assert_called_once_with(config=config, data_location=None)
                    assert result == mock_dataset_instance

    def test_setup_dataset_with_both_keys_present(self):
        """Test normal case where both dataset_class and data_location are present."""
        # Create a complete config
        config = {
            "model_inputs": {
                "test_dataset": {
                    "dataset_class": "HyraxRandomIterableDataset",
                    "data_location": "/some/valid/path",
                }
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
                    mock_dataset_cls.assert_called_once_with(config=config, data_location="/some/valid/path")
                    assert result == mock_dataset_instance

    def test_setup_dataset_sets_tensorboardx_logger(self):
        """Test that tensorboardx_logger is properly set on the dataset."""
        # Create a complete config
        config = {
            "model_inputs": {
                "test_dataset": {
                    "dataset_class": "HyraxRandomIterableDataset",
                    "data_location": "/some/valid/path",
                }
            }
        }

        # Mock dataset class, registry, and logger
        mock_dataset_instance = MagicMock()
        mock_dataset_cls = MagicMock(return_value=mock_dataset_instance)
        mock_logger = MagicMock()

        with patch("hyrax.data_sets.data_provider.generate_data_request_from_config") as mock_generate:
            with patch("hyrax.pytorch_ignite.is_iterable_dataset_requested") as mock_is_iterable:
                with patch("hyrax.pytorch_ignite.fetch_dataset_class") as mock_fetch_cls:
                    # Set up mocks
                    mock_generate.return_value = config["model_inputs"]
                    mock_is_iterable.return_value = True
                    mock_fetch_cls.return_value = mock_dataset_cls

                    # Call the function with logger
                    result = setup_dataset(config, tensorboardx_logger=mock_logger)

                    # Verify the logger was set on the dataset
                    assert result.tensorboardx_logger == mock_logger
