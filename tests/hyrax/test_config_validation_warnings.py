"""Tests for configuration validation warnings.

This module tests that ConfigManager properly warns users when data_request
configuration fails Pydantic validation.
"""

import logging
import warnings

from hyrax.config_utils import ConfigManager


def test_set_config_warns_on_invalid_data_request(caplog):
    """ConfigManager.set_config logs warning when data_request validation fails."""

    cm = ConfigManager()
    # Invalid: flat dict without a friendly name — no "dataset_class" nested inside
    # a named sub-key.  Fails DataRequestDefinition validation with a
    # "friendly name" error.
    invalid_dict = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/dev/null",
            "fields": ["image"],
            # dataset_class at top level — friendly name is missing
        }
    }

    with caplog.at_level(logging.WARNING):
        cm.set_config("data_request", invalid_dict)

    # Should log a warning
    assert "Configuration for 'data_request' failed Pydantic validation" in caplog.text
    assert "friendly name" in caplog.text
    assert "will be used as-is" in caplog.text

    # Invalid data is still stored as-is (backward compatibility)
    rendered = cm.config["data_request"]
    assert rendered == invalid_dict


def test_set_config_no_warning_on_valid_data_request(caplog):
    """ConfigManager.set_config does not warn when validation succeeds."""

    cm = ConfigManager()
    # Valid configuration — friendly name "data" explicitly provided
    valid_dict = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "somewhere",
                "fields": ["image"],
                "primary_id_field": "object_id",
            }
        }
    }

    with caplog.at_level(logging.WARNING):
        cm.set_config("data_request", valid_dict)

    # Should not log a warning about validation failure
    assert "failed Pydantic validation" not in caplog.text

    # Valid data is validated and coerced
    rendered = cm.config["data_request"]
    assert "train" in rendered
    # Friendly name "data" is preserved verbatim
    assert rendered["train"]["data"]["dataset_class"] == "HyraxRandomDataset"


def test_init_warns_on_invalid_data_request_in_toml(caplog, tmp_path):
    """ConfigManager.__init__ logs warning when TOML has invalid data_request."""

    # Create a TOML file with invalid data_request — flat structure without a
    # friendly name sub-key (dataset_class directly under train).
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(
        """
[general]
dev_mode = true

[data_request.train]
dataset_class = "HyraxRandomDataset"
data_location = "/dev/null"
fields = ["image"]
# dataset_class at group level — friendly name sub-key is missing
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
    assert "friendly name" in caplog.text
    assert "will be used as-is" in caplog.text

    # Invalid config is still loaded as-is
    assert "data_request" in cm.config
    assert "train" in cm.config["data_request"]


def test_init_no_warning_on_valid_data_request_in_toml(caplog, tmp_path):
    """ConfigManager.__init__ does not warn when TOML has valid data_request."""

    # Create a TOML file with valid data_request — friendly name sub-key "data"
    # is explicitly provided under [data_request.train.data].
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(
        """
[general]
dev_mode = true

[data_request.train.data]
dataset_class = "HyraxRandomDataset"
data_location = "/dev/null"
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


def test_init_migrates_legacy_model_inputs_to_data_request(caplog, tmp_path):
    """Legacy configs with [model_inputs] are auto-migrated to [data_request].

    The v1 → v2 migration in ``hyrax.config_migrations`` renames the legacy
    table, emits a ``DeprecationWarning``, and then the unchanged Pydantic
    validation runs against ``data_request`` as usual.
    """

    # Legacy-style config: no config_version, uses the old [model_inputs] name,
    # and the subgroup is invalid (friendly name sub-key missing).
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(
        """
config_version = 1
[general]
dev_mode = true

[model_inputs.train]
dataset_class = "HyraxRandomDataset"
data_location = "/dev/null"
# dataset_class at group level — friendly name is missing → triggers warning
"""
    )

    # Minimal default config
    default_config_file = tmp_path / "default_config.toml"
    default_config_file.write_text(
        """
[general]
dev_mode = false
"""
    )

    with caplog.at_level(logging.WARNING), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cm = ConfigManager(
            runtime_config_filepath=str(config_file),
            default_config_filepath=str(default_config_file),
        )

    # Migration emits a DeprecationWarning naming [model_inputs].
    assert any(
        issubclass(w.category, DeprecationWarning) and "model_inputs" in str(w.message) for w in caught
    ), "Expected a DeprecationWarning about [model_inputs]"

    # After migration, the legacy key is gone and data_request carries the content.
    assert "model_inputs" not in cm.config
    assert "data_request" in cm.config
    assert "train" in cm.config["data_request"]

    # Pydantic still flags the broken subgroup on the migrated content.
    assert "Configuration loaded from TOML has 'data_request' that failed Pydantic validation" in caplog.text


def test_init_migrates_up_to_date_legacy_model_inputs_to_data_request(caplog, tmp_path):
    """Legacy configs with [data_request] but no config_version are migrated correctly.

    The v1 → v2 migration in ``hyrax.config_migrations`` works as expected even
    if the legacy config has the correct table name.
    """

    # Legacy-style config: no config_version, uses the new [data_request] name,
    # and the subgroup is invalid (friendly name sub-key missing).
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(
        """
[general]
dev_mode = true

[data_request.train]
dataset_class = "HyraxRandomDataset"
data_location = "/dev/null"
# dataset_class at group level — friendly name is missing → triggers warning
"""
    )

    # Minimal default config
    default_config_file = tmp_path / "default_config.toml"
    default_config_file.write_text(
        """
config_version = 2

[general]
dev_mode = false
"""
    )

    cm = ConfigManager(
        runtime_config_filepath=str(config_file),
        default_config_filepath=str(default_config_file),
    )

    # After migration, the legacy key is gone and data_request carries the content.
    assert "model_inputs" not in cm.config
    assert "data_request" in cm.config
    assert "train" in cm.config["data_request"]
