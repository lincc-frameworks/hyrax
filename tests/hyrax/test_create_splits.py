"""Unit tests for the CreateSplits verb (src/hyrax/verbs/create_splits.py).

These tests exercise ``CreateSplits.run()`` itself -- results directory
creation, persisted split artifacts, and the returned DataProvider mapping --
as opposed to tests/hyrax/test_splitting_utils.py, which covers the
lower-level ``splitting_utils.create_splits`` function that ``run()`` calls.
"""

import numpy as np
import pytest
import tomlkit

import hyrax
from hyrax.datasets.data_provider import DataProvider
from hyrax.verbs.create_splits import CreateSplits


def _make_config(tmp_path, *, split=None, balance=None, size=100, seed=24601, groups=("train",)):
    """Return a Hyrax config wired to HyraxRandomDataset with a data_request
    for each requested group, all sharing the same data_location."""
    h = hyrax.Hyrax()
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["data_set"]["HyraxRandomDataset"]["size"] = size
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = seed
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    if split is not None:
        h.config["split"] = split
    if balance is not None:
        h.config["balance"] = balance

    data_location = str(tmp_path / "data")
    h.config["data_request"] = {
        g: {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": data_location,
                "primary_id_field": "object_id",
            }
        }
        for g in groups
    }
    return h.config


# ---------------------------------------------------------------------------
# Results directory creation
# ---------------------------------------------------------------------------


def test_run_creates_a_splits_results_directory(tmp_path):
    """run() creates exactly one new '*-splits-*' directory under results_dir."""
    config = _make_config(tmp_path, size=20)
    verb = CreateSplits(config)

    verb.run()

    split_dirs = [p for p in tmp_path.glob("*-splits-*") if p.is_dir()]
    assert len(split_dirs) == 1


def test_run_twice_creates_two_distinct_results_directories(tmp_path):
    """Each call to run() creates its own timestamped results directory."""
    config = _make_config(tmp_path, size=20)
    verb = CreateSplits(config)

    verb.run()
    verb.run()

    split_dirs = [p for p in tmp_path.glob("*-splits-*") if p.is_dir()]
    assert len(split_dirs) == 2


# ---------------------------------------------------------------------------
# Persisted artifacts
# ---------------------------------------------------------------------------


def test_run_persists_split_npz_and_config_toml(tmp_path):
    """run() writes <group>_split.npz and split_config.toml into the results dir."""
    split_cfg = {"train": 0.6, "validate": 0.4, "rng_seed": 1}
    config = _make_config(tmp_path, split=split_cfg, size=50, groups=("train", "validate"))
    verb = CreateSplits(config)

    verb.run()

    results_dir = next(p for p in tmp_path.glob("*-splits-*") if p.is_dir())
    assert (results_dir / "train_split.npz").exists()
    assert (results_dir / "validate_split.npz").exists()
    assert (results_dir / "split_config.toml").exists()

    train_npz = np.load(results_dir / "train_split.npz")
    assert len(train_npz["indexes"]) == 30
    validate_npz = np.load(results_dir / "validate_split.npz")
    assert len(validate_npz["indexes"]) == 20


def test_run_also_logs_runtime_config(tmp_path):
    """run() calls log_runtime_config, writing runtime_config.toml alongside the splits."""
    config = _make_config(tmp_path, size=20)
    verb = CreateSplits(config)

    verb.run()

    results_dir = next(p for p in tmp_path.glob("*-splits-*") if p.is_dir())
    runtime_toml = results_dir / "runtime_config.toml"
    assert runtime_toml.exists()
    parsed = tomlkit.parse(runtime_toml.read_text())
    assert "data_request" in parsed


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------


def test_run_returns_dict_of_data_providers(tmp_path):
    """run() returns a dict keyed by data group, with DataProvider instances."""
    config = _make_config(tmp_path, size=20, groups=("train",))
    verb = CreateSplits(config)

    result = verb.run()

    assert isinstance(result, dict)
    assert set(result.keys()) == {"train"}
    assert isinstance(result["train"], DataProvider)


def test_run_assigns_split_indices_onto_returned_providers(tmp_path):
    """The DataProvider objects returned by run() carry split_indices sized per config['split']."""
    split_cfg = {"train": 0.7, "validate": 0.3, "rng_seed": 3}
    config = _make_config(tmp_path, split=split_cfg, size=100, groups=("train", "validate"))
    verb = CreateSplits(config)

    result = verb.run()

    assert result["train"].split_indices is not None
    assert result["validate"].split_indices is not None
    assert len(result["train"].split_indices) == 70
    assert len(result["validate"].split_indices) == 30

    train_set = set(result["train"].split_indices)
    validate_set = set(result["validate"].split_indices)
    assert train_set.isdisjoint(validate_set)


def test_run_sets_split_weights_when_balance_configured(tmp_path):
    """When config['balance'] targets a group, run() assigns split_weights onto it."""
    h = hyrax.Hyrax()
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 100
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 24601
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]
    h.config["data_set"]["HyraxRandomDataset"]["provided_labels"] = ["A", "B"]
    h.config["split"] = {"train": 0.7, "validate": 0.3, "rng_seed": 4}
    h.config["balance"] = {"field": "label", "groups": ["train"], "distribution": {}}
    data_location = str(tmp_path / "data")
    h.config["data_request"] = {
        g: {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": data_location,
                "primary_id_field": "object_id",
            }
        }
        for g in ("train", "validate")
    }

    verb = CreateSplits(h.config)
    result = verb.run()

    assert result["train"].split_weights is not None
    assert result["validate"].split_weights is None


# ---------------------------------------------------------------------------
# run_cli()
# ---------------------------------------------------------------------------


def test_run_cli_delegates_to_run(tmp_path):
    """run_cli() invokes run(), producing the same on-disk side effects."""
    config = _make_config(tmp_path, size=20)
    verb = CreateSplits(config)

    verb.run_cli()

    split_dirs = [p for p in tmp_path.glob("*-splits-*") if p.is_dir()]
    assert len(split_dirs) == 1


def test_run_cli_accepts_and_ignores_args(tmp_path):
    """run_cli() accepts an args object (unused) without raising."""
    config = _make_config(tmp_path, size=20)
    verb = CreateSplits(config)

    class MockArgs:
        pass

    verb.run_cli(MockArgs())

    split_dirs = [p for p in tmp_path.glob("*-splits-*") if p.is_dir()]
    assert len(split_dirs) == 1


# ---------------------------------------------------------------------------
# Verb metadata / registration
# ---------------------------------------------------------------------------


def test_verb_metadata():
    """CreateSplits declares the expected cli_name and empty data-group requirements."""
    assert CreateSplits.cli_name == "create_splits"
    assert CreateSplits.REQUIRED_DATA_GROUPS == ()
    assert CreateSplits.OPTIONAL_DATA_GROUPS == ()


def test_run_does_not_require_data_request_validation(tmp_path):
    """With no REQUIRED/OPTIONAL_DATA_GROUPS, __init__ does not raise even for
    a config whose data_request would fail validation for other verbs."""
    config = _make_config(tmp_path, size=10, groups=("train",))
    # Should not raise, since CreateSplits declares no required/optional groups.
    CreateSplits(config)


def test_setup_parser_adds_no_arguments():
    """setup_parser() is a no-op: parsing an empty argument list succeeds."""
    import argparse

    parser = argparse.ArgumentParser()
    CreateSplits.setup_parser(parser)

    args = parser.parse_args([])
    assert vars(args) == {}


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_run_propagates_invalid_split_config_error(tmp_path):
    """run() surfaces RuntimeError from splitting_utils validation (fractions > 1.0)
    without creating persisted split files."""
    split_cfg = {"train": 0.8, "validate": 0.5}
    config = _make_config(tmp_path, split=split_cfg, size=50, groups=("train", "validate"))
    verb = CreateSplits(config)

    with pytest.raises(RuntimeError, match="sum to"):
        verb.run()

    results_dir = next(p for p in tmp_path.glob("*-splits-*") if p.is_dir())
    assert not (results_dir / "split_config.toml").exists()
