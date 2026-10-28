"""Tests for migration 007: removal of KafkaStreamDataset.peek_timeout."""

import tomlkit

from hyrax.config_migrations import CURRENT_CONFIG_VERSION, migrate_config


def test_migrate_007_removes_peek_timeout():
    """peek_timeout is dropped and the rest of the Kafka table is untouched."""
    cfg = tomlkit.parse(
        "config_version = 7\n"
        "[data_set.KafkaStreamDataset]\n"
        'topics = ["alerts"]\n'
        "peek_timeout = 120.0\n"
        "poll_timeout = 1.0\n"
    )

    migrated = migrate_config(cfg)

    kafka = migrated["data_set"]["KafkaStreamDataset"]
    assert "peek_timeout" not in kafka
    assert kafka["topics"] == ["alerts"]
    assert kafka["poll_timeout"] == 1.0
    assert migrated["config_version"] == CURRENT_CONFIG_VERSION


def test_migrate_007_without_kafka_table_is_noop():
    """A v7 config that never configured Kafka passes through cleanly."""
    cfg = tomlkit.parse("config_version = 7\n[data_set]\nseed = 42\n")

    migrated = migrate_config(cfg)

    assert migrated["data_set"]["seed"] == 42
    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
