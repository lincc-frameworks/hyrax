"""Tests for migration 005: preload_cache/preload_threads → data_loader.num_workers."""

import tomlkit

from hyrax.config_migrations import CURRENT_CONFIG_VERSION, migrate_config


def test_migrate_005_custom_preload_threads_becomes_num_workers():
    """A user who set preload_threads to a non-default value gets num_workers."""
    cfg = tomlkit.parse("config_version = 5\n[data_set]\npreload_cache = true\npreload_threads = 8\n")

    migrated = migrate_config(cfg)

    assert "preload_cache" not in migrated["data_set"]
    assert "preload_threads" not in migrated["data_set"]
    assert migrated["data_loader"]["num_workers"] == 8


def test_migrate_005_custom_threads_adds_to_existing_num_workers():
    """preload_threads is added to an already-set num_workers."""
    cfg = tomlkit.parse(
        "config_version = 5\n"
        "[data_set]\n"
        "preload_cache = true\n"
        "preload_threads = 4\n"
        "[data_loader]\n"
        "num_workers = 2\n"
    )

    migrated = migrate_config(cfg)

    assert migrated["data_loader"]["num_workers"] == 6


def test_migrate_005_default_50_threads_dropped():
    """The old HPC-tuned default of 50 is treated as unset and not migrated."""
    cfg = tomlkit.parse("config_version = 5\n[data_set]\npreload_cache = true\npreload_threads = 50\n")

    migrated = migrate_config(cfg)

    assert "preload_cache" not in migrated["data_set"]
    assert "preload_threads" not in migrated["data_set"]
    assert "data_loader" not in migrated or "num_workers" not in migrated.get("data_loader", {})


def test_migrate_005_preload_cache_false_dropped():
    """When preload_cache was off, both keys are just removed."""
    cfg = tomlkit.parse("config_version = 5\n[data_set]\npreload_cache = false\npreload_threads = 8\n")

    migrated = migrate_config(cfg)

    assert "preload_cache" not in migrated["data_set"]
    assert "preload_threads" not in migrated["data_set"]
    assert "data_loader" not in migrated or "num_workers" not in migrated.get("data_loader", {})


def test_migrate_005_no_preload_keys_is_noop():
    """A v5 config without preload keys passes through cleanly."""
    cfg = tomlkit.parse("config_version = 5\n[data_set]\nseed = 42\n")

    migrated = migrate_config(cfg)

    assert migrated["data_set"]["seed"] == 42
    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
