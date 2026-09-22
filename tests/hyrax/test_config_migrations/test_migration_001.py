"""Tests for migration 001: rename [model_inputs] → [data_request]."""

import logging
import warnings

import tomlkit

from hyrax.config_migrations import CURRENT_CONFIG_VERSION, migrate_config


def test_migrate_config_legacy_model_inputs_warns_and_renames(caplog):
    """A v1-era config (no config_version, uses [model_inputs]) is upgraded."""
    cfg = tomlkit.parse("config_version = 1\n[model_inputs]\ntrain = 1\n")

    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        migrated = migrate_config(cfg)

    assert "model_inputs" not in migrated
    assert migrated["data_request"]["train"] == 1
    assert migrated["config_version"] == CURRENT_CONFIG_VERSION

    assert any(
        issubclass(w.category, DeprecationWarning) and "model_inputs" in str(w.message) for w in caught
    )
    assert "model_inputs" in caplog.text
