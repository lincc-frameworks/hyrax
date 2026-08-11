"""Tests for migration 004: move [umap] → [reduce]."""

import warnings

import tomlkit

from hyrax.config_migrations import CURRENT_CONFIG_VERSION, migrate_config


def test_migrate_config_004_moves_umap_to_reduce():
    """A v4 config migrates legacy umap sections into [reduce] and adding tsne, pca."""
    cfg = tomlkit.parse(
        "config_version = 4\n"
        "[umap]\n"
        "fit_sample_size = 1024\n"
        'model_path = "some_path"\n'
        "save_fit_umap = true\n"
        "parallel = false\n"
        'name = "umap.UMAP"\n'
        "[umap.UMAP]\n"
        "n_components = 2\n"
        "n_neighbors = 15\n"
    )

    migrated = migrate_config(cfg)

    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
    assert "umap" not in migrated
    assert "umap.UMAP" not in migrated

    assert migrated["reduce"]["umap"]["fit_sample_size"] == 1024
    assert migrated["reduce"]["umap"]["model_path"] == "some_path"
    assert migrated["reduce"]["umap"]["kwargs"]["n_components"] == 2
    assert migrated["reduce"]["umap"]["kwargs"]["n_neighbors"] == 15
    assert migrated["reduce"]["save_fit_model"] is True
    assert migrated["reduce"]["parallel"] is False
    assert migrated["reduce"]["algorithm"] == "umap"
    assert migrated["reduce"]["batch_size"] == 1024

    assert migrated["reduce"]["tsne"]["kwargs"]["n_components"] == 2
    assert migrated["reduce"]["tsne"]["kwargs"]["perplexity"] == 30.0

    assert migrated["reduce"]["pca"]["fit_sample_size"] == 1024
    assert migrated["reduce"]["pca"]["model_path"] is False
    assert migrated["reduce"]["pca"]["kwargs"]["n_components"] == 2


def test_migrate_config_004_noop_when_reduce_already_current():
    """A current config with [reduce] already present is left unchanged."""
    cfg = tomlkit.parse(
        f"config_version = {CURRENT_CONFIG_VERSION}\n"
        "[reduce]\n"
        'algorithm = "umap"\n'
        "[reduce.umap]\n"
        "fit_sample_size = 1024\n"
        "[reduce.umap.kwargs]\n"
        "n_components = 2\n"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        migrated = migrate_config(cfg)

    assert migrated["config_version"] == CURRENT_CONFIG_VERSION
    assert migrated["reduce"]["algorithm"] == "umap"
    assert migrated["reduce"]["umap"]["fit_sample_size"] == 1024
    assert migrated["reduce"]["umap"]["kwargs"]["n_components"] == 2
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
