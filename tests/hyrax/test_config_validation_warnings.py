"""Tests for configuration validation warnings.

This module tests that ConfigManager properly warns users when data_request
or model_inputs configuration fails Pydantic validation.
"""

import logging

from hyrax.config_utils import ConfigManager


def test_set_config_warns_on_invalid_data_request(caplog):
    """ConfigManager.set_config logs warning when data_request validation fails."""

    cm = ConfigManager()
    # Invalid: missing primary_id_field required by DataRequestDefinition validation
    invalid_dict = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "fields": ["image"],
            # Missing primary_id_field - fails DataRequestDefinition validation
        }
    }

    with caplog.at_level(logging.WARNING):
        cm.set_config("data_request", invalid_dict)

    # Should log a warning
    assert "Configuration for 'data_request' failed Pydantic validation" in caplog.text
    assert "primary_id_field" in caplog.text
    assert "will be used as-is" in caplog.text

    # Invalid data is still stored as-is (backward compatibility)
    rendered = cm.config["data_request"]
    assert rendered == invalid_dict


def test_set_config_warns_on_invalid_model_inputs(caplog):
    """ConfigManager.set_config logs warning when model_inputs validation fails."""

    cm = ConfigManager()
    # Invalid structure that can't be validated
    invalid_dict = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            # Missing primary_id_field
        }
    }

    with caplog.at_level(logging.WARNING):
        cm.set_config("model_inputs", invalid_dict)

    # Should log a warning
    assert "Configuration for 'model_inputs' failed Pydantic validation" in caplog.text
    assert "will be used as-is" in caplog.text

    # Invalid data is still stored as-is
    rendered = cm.config["model_inputs"]
    assert rendered == invalid_dict


def test_set_config_no_warning_on_valid_data_request(caplog):
    """ConfigManager.set_config does not warn when validation succeeds."""

    cm = ConfigManager()
    # Valid configuration
    valid_dict = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "fields": ["image"],
            "primary_id_field": "object_id",
        }
    }

    with caplog.at_level(logging.WARNING):
        cm.set_config("data_request", valid_dict)

    # Should not log a warning about validation failure
    assert "failed Pydantic validation" not in caplog.text

    # Valid data is validated and coerced
    rendered = cm.config["data_request"]
    assert "train" in rendered
    # After validation, structure is wrapped in 'data' key
    assert rendered["train"]["data"]["dataset_class"] == "HyraxRandomDataset"


def test_init_warns_on_invalid_data_request_in_toml(caplog, tmp_path):
    """ConfigManager.__init__ logs warning when TOML has invalid data_request."""

    # Create a TOML file with invalid data_request
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(
        """
[general]
dev_mode = true

[data_request.train]
dataset_class = "HyraxRandomDataset"
fields = ["image"]
# Missing primary_id_field - fails validation
"""
    )

    # Create a minimal default config
    default_config_file = tmp_path / "default_config.toml"
    default_config_file.write_text(
        """
[general]
dev_mode = false
"""
    )

    with caplog.at_level(logging.WARNING):
        cm = ConfigManager(
            runtime_config_filepath=str(config_file),
            default_config_filepath=str(default_config_file),
        )

    # Should log a warning
    assert "Configuration loaded from TOML has 'data_request' that failed Pydantic validation" in caplog.text
    assert "primary_id_field" in caplog.text
    assert "will be used as-is" in caplog.text

    # Invalid config is still loaded as-is
    assert "data_request" in cm.config
    assert "train" in cm.config["data_request"]


def test_init_no_warning_on_valid_data_request_in_toml(caplog, tmp_path):
    """ConfigManager.__init__ does not warn when TOML has valid data_request."""

    # Create a TOML file with valid data_request
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(
        """
[general]
dev_mode = true

[data_request.train]
dataset_class = "HyraxRandomDataset"
fields = ["image"]
primary_id_field = "object_id"
"""
    )

    # Create a minimal default config
    default_config_file = tmp_path / "default_config.toml"
    default_config_file.write_text(
        """
[general]
dev_mode = false
"""
    )

    with caplog.at_level(logging.WARNING):
        cm = ConfigManager(
            runtime_config_filepath=str(config_file),
            default_config_filepath=str(default_config_file),
        )

    # Should not log a warning about validation failure
    assert "failed Pydantic validation" not in caplog.text

    # Valid config is loaded and validated
    assert "data_request" in cm.config
    assert "train" in cm.config["data_request"]


def test_init_no_warning_when_no_data_request_in_toml(caplog, tmp_path):
    """ConfigManager.__init__ does not warn when TOML has no data_request."""

    # Create a TOML file without data_request
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(
        """
[general]
dev_mode = true

[train]
batch_size = 32
"""
    )

    # Create a minimal default config
    default_config_file = tmp_path / "default_config.toml"
    default_config_file.write_text(
        """
[general]
dev_mode = false

[train]
batch_size = 16
"""
    )

    with caplog.at_level(logging.WARNING):
        cm = ConfigManager(
            runtime_config_filepath=str(config_file),
            default_config_filepath=str(default_config_file),
        )

    # Should not log any validation warnings
    assert "failed Pydantic validation" not in caplog.text
    assert "data_request" not in cm.config


def test_completely_invalid_structure_still_warns(caplog):
    """ConfigManager.set_config warns even for completely invalid structures."""

    cm = ConfigManager()
    # Completely invalid structure
    invalid_data = {"random_key": "random_value", "nested": {"deeply": {"invalid": 123}}}

    with caplog.at_level(logging.WARNING):
        cm.set_config("data_request", invalid_data)

    # Should log a warning
    assert "Configuration for 'data_request' failed Pydantic validation" in caplog.text
    assert "will be used as-is" in caplog.text

    # Invalid data is stored as-is
    rendered = cm.config["data_request"]
    assert rendered == invalid_data


def test_init_validates_both_data_request_and_model_inputs(caplog, tmp_path):
    """ConfigManager.__init__ validates both data_request and model_inputs if both present."""

    # Create a TOML file with both invalid data_request and invalid model_inputs
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(
        """
[general]
dev_mode = true

[data_request.train]
dataset_class = "HyraxRandomDataset"
# Missing primary_id_field - should trigger warning

[model_inputs.validate]
dataset_class = "HyraxCifarDataset"
# Missing primary_id_field - should also trigger warning
"""
    )

    # Create a minimal default config
    default_config_file = tmp_path / "default_config.toml"
    default_config_file.write_text(
        """
[general]
dev_mode = false
"""
    )

    with caplog.at_level(logging.WARNING):
        cm = ConfigManager(
            runtime_config_filepath=str(config_file),
            default_config_filepath=str(default_config_file),
        )

    # Should log warnings for both keys
    assert "Configuration loaded from TOML has 'data_request' that failed Pydantic validation" in caplog.text
    assert "Configuration loaded from TOML has 'model_inputs' that failed Pydantic validation" in caplog.text
    assert caplog.text.count("primary_id_field") >= 2  # Should mention it for both

    # Both invalid configs are still loaded as-is
    assert "data_request" in cm.config
    assert "model_inputs" in cm.config
