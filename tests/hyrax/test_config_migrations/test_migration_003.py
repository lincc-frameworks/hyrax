"""Tests for migration 003: move split_fraction → [split]."""

import warnings

import pytest
import tomlkit

from hyrax.config_migrations import CURRENT_CONFIG_VERSION, migrate_config


def test_migrate_config_003_moves_split_fraction_to_split():
    """A v3 config with split_fraction in data_request moves the value to [split]."""
    cfg = tomlkit.parse(
        "config_version = 3\n"
        "[data_request.train.data]\n"
        'dataset_class = "HyraxRandomDataset"\n'
        "split_fraction = 0.7\n"
        "[data_request.validate.data]\n"
        'dataset_class = "HyraxRandomDataset"\n'
        "split_fraction = 0.3\n"
    )

    migrated = migrate_config(cfg)

    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
    # split_fraction should be gone from data_request
    assert "split_fraction" not in migrated["data_request"]["train"]["data"]
    assert "split_fraction" not in migrated["data_request"]["validate"]["data"]
    # Values should be promoted to the [split] table
    assert migrated["split"]["train"] == pytest.approx(0.7)
    assert migrated["split"]["validate"] == pytest.approx(0.3)


def test_migrate_config_003_noop_when_no_split_fraction():
    """A v4 config with no split_fraction anywhere is returned unchanged."""
    cfg = tomlkit.parse(
        f"config_version = {CURRENT_CONFIG_VERSION}\n"
        "[data_request.train.data]\n"
        'dataset_class = "HyraxRandomDataset"\n'
        "[split]\n"
        "train = 1.0\n"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        migrated = migrate_config(cfg)

    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
    assert "split_fraction" not in migrated.get("data_request", {}).get("train", {}).get("data", {})
    assert migrated["split"]["train"] == pytest.approx(1.0)
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
