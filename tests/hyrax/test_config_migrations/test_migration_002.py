"""Tests for migration 002: move data_loader.shuffle → train.shuffle."""

import tomlkit

from hyrax.config_migrations import CURRENT_CONFIG_VERSION, migrate_config


def test_migrate_config_moves_data_loader_shuffle_to_train():
    """A v2 config moves global data_loader.shuffle to train.shuffle."""
    cfg = tomlkit.parse("config_version = 2\n[data_loader]\nshuffle = false\nbatch_size = 8\n")

    migrated = migrate_config(cfg)

    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
    assert "shuffle" not in migrated["data_loader"]
    assert migrated["data_loader"]["batch_size"] == 8
    assert migrated["train"]["shuffle"] is False


def test_migrate_config_without_data_loader_shuffle_keeps_existing_train_shuffle():
    """The shuffle migration is a no-op when the legacy key is absent."""
    cfg = tomlkit.parse("config_version = 2\n[train]\nshuffle = true\n[data_loader]\nbatch_size = 8\n")

    migrated = migrate_config(cfg)

    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
    assert migrated["train"]["shuffle"] is True
    assert "shuffle" not in migrated["data_loader"]
