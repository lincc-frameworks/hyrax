"""Tests for migration 006: rename [reduce] → [reduce_dimensions]."""

import tomlkit

from hyrax.config_migrations import CURRENT_CONFIG_VERSION, migrate_config


def test_migrate_config_006_rename_reduce_to_reduce_dimensions():
    """A v6 config migrates [reduce] to [reduce_dimensions]."""
    cfg = tomlkit.parse(
        "config_version = 6\n"
        "[reduce]\n"
        'algorithm = "umap"\n'
        "[reduce.umap]\n"
        "fit_sample_size = 1024\n"
        "[reduce.umap.kwargs]\n"
        "n_components = 2\n"
        "[reduce.tsne.kwargs]\n"
        "perplexity = 30.0\n"
        "[reduce.pca]\n"
        "fit_sample_size = 1024\n"
        "[reduce.pca.kwargs]\n"
        "n_components = 2\n"
    )

    migrated = migrate_config(cfg)

    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
    assert "reduce" not in migrated
    assert "reduce_dimensions" in migrated

    assert migrated["reduce_dimensions"]["algorithm"] == "umap"
    assert migrated["reduce_dimensions"]["umap"]["fit_sample_size"] == 1024
    assert migrated["reduce_dimensions"]["umap"]["kwargs"]["n_components"] == 2
    assert migrated["reduce_dimensions"]["tsne"]["kwargs"]["perplexity"] == 30.0
    assert migrated["reduce_dimensions"]["pca"]["fit_sample_size"] == 1024
    assert migrated["reduce_dimensions"]["pca"]["kwargs"]["n_components"] == 2
