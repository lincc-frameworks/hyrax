"""Tests for the versioned config migration system."""

import logging
import warnings

import pytest
import tomlkit

from hyrax import config_migrations
from hyrax.config_migrations import (
    CURRENT_CONFIG_VERSION,
    DEPRECATED_KEY_NAMES,
    MigrationStep,
    migrate_config,
    move_key,
    rename_table,
)
from hyrax.config_utils import ConfigManager

# ---------------------------------------------------------------------------
# rename_table helper
# ---------------------------------------------------------------------------


def test_rename_table_old_only():
    """rename_table moves an isolated old table to the new name."""
    cfg = tomlkit.parse("[model_inputs]\ntrain = 1\n")
    changed = rename_table(cfg, "model_inputs", "data_request")
    assert changed is True
    assert "model_inputs" not in cfg
    assert cfg["data_request"]["train"] == 1


def test_rename_table_no_old_key_is_noop():
    """If the old table is absent, rename_table reports no change."""
    cfg = tomlkit.parse("[data_request]\ntrain = 1\n")
    assert rename_table(cfg, "model_inputs", "data_request") is False
    assert "data_request" in cfg


def test_rename_table_merges_when_both_present():
    """When both tables exist, the new key wins on leaf collisions.

    Non-conflicting leaves from the old table are preserved in the merged
    result, mirroring ConfigManager.merge_configs semantics.
    """
    cfg = tomlkit.parse(
        '[model_inputs]\ntrain = 1\nshared = "from_old"\n[data_request]\nvalidate = 2\nshared = "from_new"\n'
    )
    assert rename_table(cfg, "model_inputs", "data_request") is True
    assert "model_inputs" not in cfg
    assert cfg["data_request"]["train"] == 1
    assert cfg["data_request"]["validate"] == 2
    # Existing data_request wins on shared leaf.
    assert cfg["data_request"]["shared"] == "from_new"


# ---------------------------------------------------------------------------
# move_key helper
# ---------------------------------------------------------------------------


def test_move_key_autovivifies_parent():
    """move_key creates intermediate tables on the destination path."""
    cfg = tomlkit.parse('[general]\nold_key = "hello"\n')
    assert move_key(cfg, "general.old_key", "new_section.renamed") is True
    assert "old_key" not in cfg["general"]
    assert cfg["new_section"]["renamed"] == "hello"


def test_move_key_missing_source_is_noop():
    """move_key reports False when the source path is absent."""
    cfg = tomlkit.parse("[general]\nkept = 1\n")
    assert move_key(cfg, "general.missing", "somewhere.else") is False


# ---------------------------------------------------------------------------
# migrate_config: version detection and chain
# ---------------------------------------------------------------------------


def test_migrate_config_legacy_model_inputs_warns_and_renames(caplog):
    """A v1-era config (no config_version, uses [model_inputs]) is upgraded."""
    cfg = tomlkit.parse("[model_inputs]\ntrain = 1\n")

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


def test_migrate_config_current_version_is_noop():
    """A clean v2 config is stamped through migrate_config unchanged."""
    cfg = tomlkit.parse("config_version = 2\n[data_request]\ntrain = 1\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        migrated = migrate_config(cfg)
    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
    assert migrated["data_request"]["train"] == 1
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_migrate_config_future_version_raises():
    """A config from a newer Hyrax is refused with an upgrade hint."""
    cfg = tomlkit.parse(f"config_version = {CURRENT_CONFIG_VERSION + 5}\n")
    with pytest.raises(RuntimeError, match=r"pip install -U hyrax"):
        migrate_config(cfg)


def test_migrate_config_non_integer_version_raises():
    """A string config_version is rejected with a clear message."""
    cfg = tomlkit.parse('config_version = "two"\n')
    with pytest.raises(RuntimeError, match="config_version must be a non-boolean integer"):
        migrate_config(cfg)


def test_migrate_config_float_version_raises():
    """A float config_version is rejected — we don't silently truncate."""
    cfg = tomlkit.parse("config_version = 2.9\n")
    with pytest.raises(RuntimeError, match="config_version must be a non-boolean integer"):
        migrate_config(cfg)


def test_migrate_config_bool_version_raises():
    """A boolean config_version is rejected — bool is an int subclass in Python."""
    cfg = tomlkit.parse("config_version = true\n")
    with pytest.raises(RuntimeError, match="config_version must be a non-boolean integer"):
        migrate_config(cfg)


def test_migrate_config_zero_version_raises():
    """A zero or negative config_version is rejected with an explicit minimum."""
    cfg = tomlkit.parse("config_version = 0\n")
    with pytest.raises(RuntimeError, match="config_version must be >= 1"):
        migrate_config(cfg)


def test_migrate_config_negative_version_raises():
    """A negative config_version is rejected."""
    cfg = tomlkit.parse("config_version = -3\n")
    with pytest.raises(RuntimeError, match="config_version must be >= 1"):
        migrate_config(cfg)


def test_migrate_config_empty_document_is_passthrough():
    """An empty user config (no file case) is returned unchanged."""
    cfg = tomlkit.document()
    migrated = migrate_config(cfg)
    # An empty document means "use defaults entirely" — no version stamp needed.
    assert len(migrated) == 0


def test_migrate_config_is_idempotent():
    """Running migrate_config twice is safe and does not re-warn."""
    cfg = tomlkit.parse("[model_inputs]\ntrain = 1\n")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        first = migrate_config(cfg)

    with warnings.catch_warnings(record=True) as caught_second:
        warnings.simplefilter("always")
        second = migrate_config(first)

    assert second["data_request"]["train"] == 1
    assert second["config_version"] == CURRENT_CONFIG_VERSION
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught_second)


def test_migrate_config_missing_migration_step_raises(monkeypatch):
    """A gap in the MIGRATIONS registry surfaces as a clear runtime error."""
    monkeypatch.setattr(config_migrations, "CURRENT_CONFIG_VERSION", 5)
    monkeypatch.setattr(config_migrations, "MIGRATIONS", {1: MigrationStep(func=lambda c: c)})

    cfg = tomlkit.parse("config_version = 1\n")
    with pytest.raises(RuntimeError, match="No migration registered"):
        migrate_config(cfg)


# ---------------------------------------------------------------------------
# End-to-end: ConfigManager wiring
# ---------------------------------------------------------------------------


def test_config_manager_runs_migration_on_load(tmp_path):
    """Loading a legacy user TOML through ConfigManager migrates it and
    emits a DeprecationWarning. The final merged config contains data_request
    and no longer references model_inputs.
    """
    user_config = tmp_path / "legacy.toml"
    user_config.write_text(
        """
[general]
dev_mode = true

[model_inputs.train]
data = {dataset_class = "HyraxRandomDataset", data_location = "/tmp/data", primary_id_field = "id"}
"""
    )

    default_config = tmp_path / "default.toml"
    default_config.write_text(
        """
config_version = 2

[general]
dev_mode = false
"""
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cm = ConfigManager(
            runtime_config_filepath=str(user_config),
            default_config_filepath=str(default_config),
        )

    assert "model_inputs" not in cm.config
    assert "data_request" in cm.config
    assert "train" in cm.config["data_request"]
    assert any(
        issubclass(w.category, DeprecationWarning) and "model_inputs" in str(w.message) for w in caught
    )


# ---------------------------------------------------------------------------
# MigrationStep and DEPRECATED_KEY_NAMES
# ---------------------------------------------------------------------------


def test_deprecated_key_names_derived_from_migrations():
    """DEPRECATED_KEY_NAMES is automatically built from MigrationStep.key_renames."""
    assert "model_inputs" in DEPRECATED_KEY_NAMES
    assert DEPRECATED_KEY_NAMES["model_inputs"] == "data_request"


def test_migration_step_without_key_renames():
    """A MigrationStep with no renames still works as a migration."""
    step = MigrationStep(func=lambda c: c)
    assert step.key_renames == {}


# ---------------------------------------------------------------------------
# set_config: deprecated key warnings
# ---------------------------------------------------------------------------


def test_set_config_deprecated_key_warns(tmp_path):
    """set_config with a deprecated top-level key emits a DeprecationWarning."""
    user_config = tmp_path / "user.toml"
    user_config.write_text("[general]\ndev_mode = true\n")

    default_config = tmp_path / "default.toml"
    default_config.write_text(
        "config_version = 2\n\n[general]\ndev_mode = false\n\n[model_inputs]\ntrain = {}\n"
    )

    cm = ConfigManager(
        runtime_config_filepath=str(user_config),
        default_config_filepath=str(default_config),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cm.set_config("model_inputs", {"train": {"data": "test"}})

    assert any(
        issubclass(w.category, DeprecationWarning)
        and "model_inputs" in str(w.message)
        and "data_request" in str(w.message)
        for w in caught
    )


def test_set_config_current_key_no_warn(tmp_path):
    """set_config with the current key name does not emit a DeprecationWarning."""
    user_config = tmp_path / "user.toml"
    user_config.write_text("[general]\ndev_mode = true\n")

    default_config = tmp_path / "default.toml"
    default_config.write_text(
        "config_version = 2\n\n[general]\ndev_mode = false\n\n[data_request]\ntrain = {}\n"
    )

    cm = ConfigManager(
        runtime_config_filepath=str(user_config),
        default_config_filepath=str(default_config),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cm.set_config("data_request", {"train": {"data": "test"}})

    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
