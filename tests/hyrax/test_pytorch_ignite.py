from unittest.mock import MagicMock, patch

import pytest

from hyrax.pytorch_ignite import setup_dataset


class TestSetupDataset:
    """Tests for the setup_dataset function."""

    def test_setup_dataset_missing_dataset_class_raises_error(self):
        """Test that missing dataset_class raises appropriate RuntimeError."""
        # Create a minimal config that would trigger iterable dataset path
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
