"""Tests for data_location path resolution in DataProvider"""

from pathlib import Path

from hyrax import Hyrax
from hyrax.data_sets.data_provider import DataProvider


def test_relative_path_resolution():
    """Test that relative data_location paths are resolved to absolute paths."""
    h = Hyrax()

    # Configure with a relative path
    data_request = {
        "test_dataset": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "./relative/path/to/data",
            "primary_id_field": "object_id",
            "fields": ["image"],
        }
    }

    h.config["data_request"] = data_request

    # Create DataProvider
    dp = DataProvider(h.config, data_request)

    # Verify the path was resolved to an absolute path in the data_request
    resolved_path = dp.data_request["test_dataset"]["data_location"]
    assert Path(resolved_path).is_absolute(), f"Path should be absolute but got: {resolved_path}"
    assert "relative/path/to/data" in resolved_path

    # Verify the path was also written back to the main config
    config_path = h.config["data_request"]["test_dataset"]["data_location"]
    assert Path(config_path).is_absolute(), f"Config path should be absolute but got: {config_path}"
    assert config_path == resolved_path


def test_absolute_path_unchanged():
    """Test that absolute data_location paths remain unchanged."""
    h = Hyrax()

    # Use an absolute path
    absolute_path = "/absolute/path/to/data"
    data_request = {
        "test_dataset": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": absolute_path,
            "primary_id_field": "object_id",
            "fields": ["image"],
        }
    }

    h.config["data_request"] = data_request

    # Create DataProvider
    dp = DataProvider(h.config, data_request)

    # Verify the path is still absolute and matches
    resolved_path = dp.data_request["test_dataset"]["data_location"]
    assert Path(resolved_path).is_absolute()
    assert resolved_path == absolute_path

    # Verify the config was updated
    config_path = h.config["data_request"]["test_dataset"]["data_location"]
    assert config_path == absolute_path


def test_none_data_location():
    """Test that None data_location values are handled correctly."""
    h = Hyrax()

    # Some datasets don't require data_location (e.g., HyraxRandomDataset)
    data_request = {
        "test_dataset": {
            "dataset_class": "HyraxRandomDataset",
            # data_location is not provided
            "primary_id_field": "object_id",
            "fields": ["image"],
        }
    }

    h.config["data_request"] = data_request

    # Create DataProvider - should not fail
    dp = DataProvider(h.config, data_request)

    # Verify data_location is None or not present
    assert dp.data_request["test_dataset"].get("data_location") is None


def test_tilde_path_expansion():
    """Test that paths with ~ are expanded correctly."""
    h = Hyrax()

    # Use a path with tilde
    tilde_path = "~/path/to/data"
    data_request = {
        "test_dataset": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": tilde_path,
            "primary_id_field": "object_id",
            "fields": ["image"],
        }
    }

    h.config["data_request"] = data_request

    # Create DataProvider
    dp = DataProvider(h.config, data_request)

    # Verify the tilde was expanded
    resolved_path = dp.data_request["test_dataset"]["data_location"]
    assert "~" not in resolved_path
    assert Path(resolved_path).is_absolute()
    assert resolved_path.startswith(str(Path.home()))


def test_multiple_datasets_path_resolution():
    """Test that multiple datasets have their paths resolved independently."""
    h = Hyrax()

    data_request = {
        "dataset_1": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "./data1",
            "primary_id_field": "object_id",
            "fields": ["image"],
        },
        "dataset_2": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "./data2",
            "fields": ["image"],
        },
    }

    h.config["data_request"] = data_request

    # Create DataProvider
    dp = DataProvider(h.config, data_request)

    # Verify both paths were resolved
    path_1 = dp.data_request["dataset_1"]["data_location"]
    path_2 = dp.data_request["dataset_2"]["data_location"]

    assert Path(path_1).is_absolute()
    assert Path(path_2).is_absolute()
    assert "data1" in path_1
    assert "data2" in path_2
    assert path_1 != path_2

    # Verify both were written back to config
    assert h.config["data_request"]["dataset_1"]["data_location"] == path_1
    assert h.config["data_request"]["dataset_2"]["data_location"] == path_2


def test_persisted_config_has_resolved_paths(tmp_path):
    """Test that the persisted runtime_config.toml contains resolved paths."""
    from hyrax.config_utils import log_runtime_config

    h = Hyrax()

    # Configure with relative path
    data_request = {
        "test_dataset": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "./my_data",
            "primary_id_field": "object_id",
            "fields": ["image"],
        }
    }

    h.config["data_request"] = data_request

    # Create DataProvider to trigger path resolution
    DataProvider(h.config, data_request)

    # Save the config
    log_runtime_config(h.config, tmp_path)

    # Read back the saved config
    import tomlkit

    saved_config_path = tmp_path / "runtime_config.toml"
    with open(saved_config_path, "r") as f:
        saved_config = tomlkit.load(f)

    # Verify the saved config has an absolute path
    saved_path = saved_config["data_request"]["test_dataset"]["data_location"]
    assert Path(saved_path).is_absolute()
    assert "my_data" in saved_path
