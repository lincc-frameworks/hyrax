"""Tests for splitting_utils: create_splits, validate_split_config, persist/load."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pytest

import hyrax

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    size: int = 100,
    seed: int = 42,
    provided_labels: list | None = None,
    split: dict | None = None,
    balance: dict | None = None,
    tmp_path: Path,
    data_location: str | None = None,
    groups: tuple[str, ...] = ("train",),
) -> dict:
    """Build a minimal Hyrax config wired to HyraxRandomDataset.

    Includes a minimal data_request so the config is ready for DataProvider.
    All groups share the same data_location unless overridden.
    """
    h = hyrax.Hyrax()
    h.config["data_set"]["HyraxRandomDataset"]["size"] = size
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = seed
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 4]
    if provided_labels is not None:
        h.config["data_set"]["HyraxRandomDataset"]["provided_labels"] = provided_labels

    if split is not None:
        h.config["split"] = split
    if balance is not None:
        h.config["balance"] = balance

    h.config["general"]["results_dir"] = str(tmp_path)
    loc = data_location or str(tmp_path)

    dr: dict = {}
    for g in groups:
        dr[g] = {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": loc,
                "primary_id_field": "object_id",
            }
        }
    h.config["data_request"] = dr

    return h.config


def _make_providers(config: dict, groups: tuple[str, ...]) -> dict:
    """Create DataProvider instances for the requested groups."""
    from hyrax.pytorch_ignite import setup_dataset

    return setup_dataset(config, splits=groups)


# ---------------------------------------------------------------------------
# Test 1: defaults reproduce current behavior (all indices, no weights)
# ---------------------------------------------------------------------------


def test_create_splits_defaults_all_indices(tmp_path):
    """Default config (fractions all 1.0, no balance) returns all indices
    in order and with None weights for every group."""
    from hyrax.splitting_utils import create_splits

    config = _make_config(size=50, tmp_path=tmp_path, groups=("train",))
    datasets = _make_providers(config, ("train",))

    result = create_splits(config, datasets)

    assert "train" in result
    train_data = result["train"]
    assert sorted(train_data["indexes"].tolist()) == list(range(50))
    assert train_data["weights"] is None
    # split_indices was assigned onto the provider
    assert datasets["train"].split_indices is not None
    assert len(datasets["train"].split_indices) == 50


# ---------------------------------------------------------------------------
# Test 2: custom fractions → non-overlapping, deterministic
# ---------------------------------------------------------------------------


def test_create_splits_custom_fractions_non_overlapping_deterministic(tmp_path):
    """train=0.7, validate=0.2, test=0.1 on same data_location → non-overlapping
    subsets of expected sizes; identical on repeated calls with same rng_seed."""
    from hyrax.splitting_utils import create_splits

    size = 100
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.7, "validate": 0.2, "test": 0.1, "rng_seed": 7}
    config = _make_config(
        size=size,
        tmp_path=tmp_path,
        split=split_cfg,
        data_location=shared_loc,
        groups=("train", "validate", "test"),
    )
    datasets = _make_providers(config, ("train", "validate", "test"))

    result = create_splits(config, datasets)

    train_idx = set(result["train"]["indexes"].tolist())
    val_idx = set(result["validate"]["indexes"].tolist())
    test_idx = set(result["test"]["indexes"].tolist())

    # Non-overlapping
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)

    # Approximate sizes (round to nearest)
    assert abs(len(train_idx) - 70) <= 1
    assert abs(len(val_idx) - 20) <= 1
    assert abs(len(test_idx) - 10) <= 1

    # Deterministic: second call with fresh providers yields identical splits
    datasets2 = _make_providers(config, ("train", "validate", "test"))
    result2 = create_splits(config, datasets2)
    assert sorted(result["train"]["indexes"].tolist()) == sorted(result2["train"]["indexes"].tolist())
    assert sorted(result["validate"]["indexes"].tolist()) == sorted(result2["validate"]["indexes"].tolist())


# ---------------------------------------------------------------------------
# Test 2a: small dataset
# ---------------------------------------------------------------------------


def test_create_splits_small_dataset(tmp_path):
    """create_splits works correctly on a very small dataset (5 items).

    Verifies split indices are assigned, non-overlapping, within bounds, and
    cover the whole dataset when fractions sum to 1.0.
    """
    from hyrax.splitting_utils import create_splits

    size = 5
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.6, "validate": 0.4, "rng_seed": 1}
    config = _make_config(
        size=size,
        tmp_path=tmp_path,
        split=split_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    result = create_splits(config, datasets)

    train_idx = result["train"]["indexes"].tolist()
    val_idx = result["validate"]["indexes"].tolist()

    assert all(0 <= i < size for i in train_idx)
    assert all(0 <= i < size for i in val_idx)
    assert set(train_idx).isdisjoint(set(val_idx))
    assert sorted(train_idx + val_idx) == list(range(size))


# ---------------------------------------------------------------------------
# Test 2b: clamping prevents overrun
# ---------------------------------------------------------------------------


def test_create_splits_clamping_prevents_overrun(tmp_path):
    """A 50/50 split of a 3-item dataset never assigns more than 3 total indices.

    round(3 × 0.5) = 2 due to banker's rounding.  Without the per-group clamp
    ``min(round(n * frac), n - offset)`` and the last-group rule, both groups
    would claim 2 indices each (total 4), overrunning the 3-item dataset.
    """
    from hyrax.splitting_utils import create_splits

    size = 3
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.5, "validate": 0.5, "rng_seed": 1}
    config = _make_config(
        size=size,
        tmp_path=tmp_path,
        split=split_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    result = create_splits(config, datasets)

    train_idx = result["train"]["indexes"].tolist()
    val_idx = result["validate"]["indexes"].tolist()

    assert len(train_idx) + len(val_idx) <= size
    assert set(train_idx).isdisjoint(set(val_idx))
    assert all(0 <= i < size for i in train_idx + val_idx)


# ---------------------------------------------------------------------------
# Test 2c: rounding leftover indices assigned to last split
# ---------------------------------------------------------------------------


def test_create_splits_rounding_leftover_indices_assigned_to_last_split(tmp_path):
    """When fractions sum to 1.0 but don't divide evenly, all leftover indices
    go to the last split group so no items are lost.

    10 items, train=0.33, validate=0.33, test=0.34:
      round(3.3)=3, round(3.3)=3, last group gets 10−6=4 (not round(3.4)=3).
    Without the last-group rule the total would be 9, silently dropping 1 item.
    """
    from hyrax.splitting_utils import create_splits

    size = 10
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.33, "validate": 0.33, "test": 0.34, "rng_seed": 1}
    config = _make_config(
        size=size,
        tmp_path=tmp_path,
        split=split_cfg,
        data_location=shared_loc,
        groups=("train", "validate", "test"),
    )
    datasets = _make_providers(config, ("train", "validate", "test"))
    result = create_splits(config, datasets)

    train_idx = result["train"]["indexes"].tolist()
    val_idx = result["validate"]["indexes"].tolist()
    test_idx = result["test"]["indexes"].tolist()

    # Every index must be assigned exactly once
    assert sorted(train_idx + val_idx + test_idx) == list(range(size))

    # train and validate get their rounded share; test absorbs the leftover
    assert len(train_idx) == 3
    assert len(val_idx) == 3
    assert len(test_idx) == 4


# ---------------------------------------------------------------------------
# Test 3: Σ > 1.0 on shared location → RuntimeError
# ---------------------------------------------------------------------------


def test_create_splits_sum_exceeds_one_raises(tmp_path):
    """train=0.8, validate=0.5 on same data_location sums to 1.3 → RuntimeError."""
    from hyrax.splitting_utils import validate_split_config

    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.8, "validate": 0.5}
    config = _make_config(
        size=50,
        tmp_path=tmp_path,
        split=split_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))

    with pytest.raises(RuntimeError, match="sum to"):
        validate_split_config(config, datasets)


# ---------------------------------------------------------------------------
# Test 4: Stratified splits preserve class proportions
# ---------------------------------------------------------------------------


def test_create_splits_stratified_proportions(tmp_path):
    """With balance.field='label', each split should contain ~equal class proportions.
    Note that this unit test introduces random selection, and thus the assertions
    are approximate (within 2% of the expected 1/3 proportion for each of 3 classes)
    rather than exact. This generally seems to be fine, but a failure here may
    indicate that the test needs to be re-run, or that the size of the dataset
    should be increased to reduce variance."""
    from hyrax.splitting_utils import create_splits

    labels = ["A", "B", "C"]
    size = 10_000
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.6, "validate": 0.2, "test": 0.2, "rng_seed": 1}
    balance_cfg = {"field": "label", "groups": [], "distribution": {}}
    config = _make_config(
        size=size,
        provided_labels=labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance=balance_cfg,
        data_location=shared_loc,
        groups=("train", "validate", "test"),
    )
    datasets = _make_providers(config, ("train", "validate", "test"))
    result = create_splits(config, datasets)

    # Build index→label mapping from the primary dataset
    primary_ds = datasets["train"].prepped_datasets["data"]
    index_to_label = {i: primary_ds.get_label(i) for i in range(size)}

    def class_fractions(indices):
        counts: dict = {}
        for idx in indices:
            label = index_to_label[idx]
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    train_fracs = class_fractions(result["train"]["indexes"].tolist())
    val_fracs = class_fractions(result["validate"]["indexes"].tolist())

    for lbl in labels:
        assert abs(train_fracs.get(lbl, 0) - 1 / 3) < 0.02, f"train class {lbl} proportion off"
        assert abs(val_fracs.get(lbl, 0) - 1 / 3) < 0.02, f"validate class {lbl} proportion off"


# ---------------------------------------------------------------------------
# Test 5: Equal rebalance – weights set for specified groups only
# ---------------------------------------------------------------------------


def test_create_splits_equal_rebalance_weights(tmp_path):
    """balance.groups=['train'] sets split_weights on train only; others remain None."""
    from hyrax.splitting_utils import create_splits

    labels = ["A", "B"]
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.7, "validate": 0.3, "rng_seed": 2}
    balance_cfg = {"field": "label", "groups": ["train"], "distribution": {}}
    config = _make_config(
        size=100,
        provided_labels=labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance=balance_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    result = create_splits(config, datasets)

    assert result["train"]["weights"] is not None, "train should have weights"
    assert result["validate"]["weights"] is None, "validate should have no weights"

    train_weights = result["train"]["weights"]
    assert len(train_weights) == len(result["train"]["indexes"])
    assert np.all(train_weights > 0)

    assert datasets["train"].split_weights is not None
    assert datasets["validate"].split_weights is None


# ---------------------------------------------------------------------------
# Test 6: Custom distribution + validation
# ---------------------------------------------------------------------------


def test_create_splits_custom_distribution(tmp_path):
    """balance.distribution with valid target fractions → weights are set;
    unknown label in distribution raises; non-unity sum raises."""
    from hyrax.splitting_utils import create_splits, validate_balance_config, validate_distribution_labels

    labels = ["cat", "dog"]
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.8, "validate": 0.2, "rng_seed": 3}
    balance_cfg = {"field": "label", "groups": ["train"], "distribution": {"cat": 0.6, "dog": 0.4}}
    config = _make_config(
        size=100,
        provided_labels=labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance=balance_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    result = create_splits(config, datasets)

    assert result["train"]["weights"] is not None

    # Unknown label in distribution raises via validate_distribution_labels
    with pytest.raises(RuntimeError, match="not found in the dataset"):
        validate_distribution_labels({"unknown_class": 1.0}, {"cat", "dog"})

    # Non-unity distribution sum raises via validate_balance_config
    bad_balance_cfg = {"field": "label", "groups": [], "distribution": {"cat": 0.6, "dog": 0.6}}
    bad_config = _make_config(
        size=100,
        provided_labels=labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance=bad_balance_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    with pytest.raises(RuntimeError, match="must sum to exactly 1.0"):
        validate_balance_config(bad_config, datasets)


# ---------------------------------------------------------------------------
# Test 6a: [label] translation — integer raw values, string aliases
# ---------------------------------------------------------------------------


def test_create_splits_label_table_translation(tmp_path):
    """[label] maps string aliases to integer raw values from get_label.

    Verifies: class_inds re-keyed to aliases; weights computed correctly;
    split_config.toml round-trips the [label] table.

    Error cases:
      - distribution key absent from [label] → RuntimeError (pre-scan)
      - dataset raw value not covered by [label] → RuntimeError (post-scan)
      - duplicate raw values in [label] → RuntimeError (pre-scan)
    """
    import tomlkit

    from hyrax.splitting_utils import (
        create_splits,
        validate_balance_config,
    )

    # HyraxRandomDataset.get_label returns integers when provided_labels=[0,1,2]
    int_labels = [0, 1, 2]
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.8, "validate": 0.2, "rng_seed": 7}
    balance_cfg = {
        "field": "label",
        "groups": ["train"],
        "distribution": {"cat": 0.5, "dog": 0.3, "bird": 0.2},
    }
    label_cfg = {"cat": 0, "dog": 1, "bird": 2}

    config = _make_config(
        size=90,
        provided_labels=int_labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance=balance_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    config["label"] = label_cfg

    datasets = _make_providers(config, ("train", "validate"))
    result = create_splits(config, datasets, results_dir=tmp_path, persist=True)

    # Weights should be set on train (it's in groups_to_balance)
    assert result["train"]["weights"] is not None
    assert len(result["train"]["weights"]) == len(result["train"]["indexes"])
    # validate group has no weights (not in groups)
    assert result["validate"]["weights"] is None

    # split_config.toml should contain [label]
    toml_path = tmp_path / "split_config.toml"
    assert toml_path.exists()
    persisted = dict(tomlkit.parse(toml_path.read_text()))
    assert "label" in persisted
    assert persisted["label"]["cat"] == 0
    assert persisted["label"]["dog"] == 1

    # Error: distribution key absent from [label] → pre-scan raise
    bad_balance_missing_key = {
        "field": "label",
        "groups": ["train"],
        "distribution": {"cat": 0.6, "unknown_alias": 0.4},
    }
    bad_config = _make_config(
        size=90,
        provided_labels=int_labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance=bad_balance_missing_key,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    bad_config["label"] = label_cfg
    bad_datasets = _make_providers(bad_config, ("train", "validate"))
    with pytest.raises(RuntimeError, match=r"not defined in \[label\]"):
        validate_balance_config(bad_config, bad_datasets)

    # Dataset raw value not covered by [label] → warning log, not an error.
    # Items with that raw value are excluded from all split groups.
    incomplete_label = {"cat": 0, "dog": 1}  # missing bird=2
    config_incomplete = _make_config(
        size=90,
        provided_labels=int_labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance={"field": "label", "groups": ["train"], "distribution": {"cat": 0.6, "dog": 0.4}},
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    config_incomplete["label"] = incomplete_label
    datasets_incomplete = _make_providers(config_incomplete, ("train", "validate"))
    result_incomplete = create_splits(config_incomplete, datasets_incomplete)  # must not raise
    # Only cat+dog items appear; bird items (raw value 2) are excluded
    all_indices = set(result_incomplete["train"]["indexes"].tolist()) | set(
        result_incomplete["validate"]["indexes"].tolist()
    )
    assert len(all_indices) > 0
    assert len(all_indices) < 90  # some items excluded (the bird ones)

    # Error: duplicate raw values in [label] → pre-scan raise
    dupe_label = {"cat": 0, "kitty": 0, "dog": 1, "bird": 2}
    config_dupe = _make_config(
        size=90,
        provided_labels=int_labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance=balance_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    config_dupe["label"] = dupe_label
    datasets_dupe = _make_providers(config_dupe, ("train", "validate"))
    with pytest.raises(RuntimeError, match="unique"):
        validate_balance_config(config_dupe, datasets_dupe)


# ---------------------------------------------------------------------------
# Test 7: Path input → load without recomputing
# ---------------------------------------------------------------------------


def test_create_splits_path_input_loads_from_file(tmp_path):
    """Persist a split, then pass paths back in config → loads from files."""
    from hyrax.splitting_utils import create_splits, persist_splits

    size = 60
    config = _make_config(size=size, tmp_path=tmp_path, groups=("train",))
    datasets = _make_providers(config, ("train",))

    # First: compute and persist
    result = create_splits(config, datasets)
    persist_splits(tmp_path, result, config)

    train_npz = tmp_path / "train_split.npz"
    assert train_npz.exists()

    # Second call: supply the path as the split value
    config2 = _make_config(size=size, tmp_path=tmp_path, split={"train": str(train_npz)}, groups=("train",))
    datasets2 = _make_providers(config2, ("train",))

    result2 = create_splits(config2, datasets2)

    assert sorted(result2["train"]["indexes"].tolist()) == sorted(result["train"]["indexes"].tolist())
    assert datasets2["train"].split_indices is not None


# ---------------------------------------------------------------------------
# Test 8: Equivalency reuse
# ---------------------------------------------------------------------------


def test_create_splits_equivalency_reuse(tmp_path):
    """Second call with identical config + results_dir reuses the persisted split;
    changing rng_seed → not equivalent."""
    from hyrax.splitting_utils import configs_equivalent, create_splits

    size = 50
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.8, "validate": 0.2, "rng_seed": 5}
    config = _make_config(
        size=size,
        tmp_path=tmp_path,
        split=split_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))

    # First call: persist
    result1 = create_splits(config, datasets, results_dir=tmp_path, persist=True)

    # Second call: identical config → finds equivalent and reloads
    datasets2 = _make_providers(config, ("train", "validate"))
    result2 = create_splits(config, datasets2, results_dir=tmp_path, persist=True)

    assert sorted(result1["train"]["indexes"].tolist()) == sorted(result2["train"]["indexes"].tolist())

    # Mutate rng_seed → should differ
    config_different = _make_config(
        size=size,
        tmp_path=tmp_path,
        split={**split_cfg, "rng_seed": 999},
        data_location=shared_loc,
        groups=("train", "validate"),
    )

    equivalent, diffs = configs_equivalent(config, config_different)
    assert not equivalent
    assert any("rng_seed" in d for d in diffs)


# ---------------------------------------------------------------------------
# Test 10: Persisted artifacts round-trip
# ---------------------------------------------------------------------------


def test_create_splits_persist_round_trip(tmp_path):
    """persist=True writes .npz files with 'indexes' array; 'weights' absent
    when None; split_config.toml is written to results_dir."""
    import tomlkit

    from hyrax.splitting_utils import create_splits

    size = 80
    labels = ["X", "Y"]
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.75, "validate": 0.25, "rng_seed": 6}
    # Use balance to trigger weight computation for train only
    balance_cfg = {"field": "label", "groups": ["train"], "distribution": {}}
    config = _make_config(
        size=size,
        provided_labels=labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance=balance_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    create_splits(config, datasets, results_dir=tmp_path, persist=True)

    train_npz_path = tmp_path / "train_split.npz"
    validate_npz_path = tmp_path / "validate_split.npz"
    split_config_toml = tmp_path / "split_config.toml"

    assert train_npz_path.exists()
    assert validate_npz_path.exists()
    assert split_config_toml.exists()

    # train has weights (balance.groups=['train']); validate does not
    train_npz = np.load(train_npz_path)
    assert "indexes" in train_npz.files
    assert "weights" in train_npz.files

    validate_npz = np.load(validate_npz_path)
    assert "indexes" in validate_npz.files
    assert "weights" not in validate_npz.files

    # split_config.toml contains relevant sections
    with open(split_config_toml) as f:
        saved_cfg = tomlkit.parse(f.read())
    assert "split" in saved_cfg
    assert "data_request" in saved_cfg


# ---------------------------------------------------------------------------
# Test 11: Path-based split → non-existent file raises RuntimeError
# ---------------------------------------------------------------------------


def test_validate_split_config_path_not_found_raises(tmp_path):
    """validate_split_config raises RuntimeError when a path-based split file does not exist."""
    from hyrax.splitting_utils import validate_split_config

    missing_path = str(tmp_path / "nonexistent_split.npz")
    config = _make_config(
        size=20,
        tmp_path=tmp_path,
        split={"train": missing_path},
        groups=("train",),
    )
    datasets = _make_providers(config, ("train",))

    with pytest.raises(RuntimeError, match="does not exist"):
        validate_split_config(config, datasets)


def test_validate_split_config_existing_path_does_not_raise(tmp_path):
    """validate_split_config accepts a path-based split value when the file exists."""
    from hyrax.splitting_utils import create_splits, persist_splits, validate_split_config

    size = 20
    config = _make_config(size=size, tmp_path=tmp_path, groups=("train",))
    datasets = _make_providers(config, ("train",))

    result = create_splits(config, datasets)
    persist_splits(tmp_path, result, config)

    train_npz = tmp_path / "train_split.npz"
    assert train_npz.exists()

    config2 = _make_config(size=size, tmp_path=tmp_path, split={"train": str(train_npz)}, groups=("train",))
    datasets2 = _make_providers(config2, ("train",))

    # Should not raise
    validate_split_config(config2, datasets2)


# ---------------------------------------------------------------------------
# Test 12: distribution value of 0.0 is now valid (range changed to [0, 1])
# ---------------------------------------------------------------------------


def test_validate_balance_config_zero_distribution_value_is_valid(tmp_path):
    """balance.distribution values of exactly 0.0 are accepted (range is [0.0, 1.0])."""
    from hyrax.splitting_utils import validate_balance_config

    labels = ["A", "B", "C"]
    # Class C gets weight 0 — valid: intentionally excluded from sampling
    balance_cfg = {"field": "label", "groups": [], "distribution": {"A": 0.7, "B": 0.3, "C": 0.0}}
    config = _make_config(
        size=90,
        provided_labels=labels,
        tmp_path=tmp_path,
        balance=balance_cfg,
        groups=("train",),
    )
    datasets = _make_providers(config, ("train",))

    # Must not raise
    validate_balance_config(config, datasets)


def test_validate_balance_config_negative_distribution_value_raises(tmp_path):
    """balance.distribution values below 0.0 are rejected."""
    from hyrax.splitting_utils import validate_balance_config

    labels = ["A", "B"]
    balance_cfg = {"field": "label", "groups": [], "distribution": {"A": 1.1, "B": -0.1}}
    config = _make_config(
        size=60,
        provided_labels=labels,
        tmp_path=tmp_path,
        balance=balance_cfg,
        groups=("train",),
    )
    datasets = _make_providers(config, ("train",))

    with pytest.raises(RuntimeError, match=r"out of range"):
        validate_balance_config(config, datasets)


def test_compute_weights_zero_distribution_gives_zero_weights(tmp_path):
    """Samples whose label has distribution=0.0 receive weight 0.0.

    A and B each target 50% of sampling; C targets 0%.  After create_splits
    every C-labelled sample in the train split must have weight 0.0, while
    every A- and B-labelled sample must have a positive weight.
    """
    from hyrax.splitting_utils import create_splits

    labels = ["A", "B", "C"]
    shared_loc = str(tmp_path / "shared_data")
    split_cfg = {"train": 0.8, "validate": 0.2, "rng_seed": 42}
    balance_cfg = {
        "field": "label",
        "groups": ["train"],
        "distribution": {"A": 0.5, "B": 0.5, "C": 0.0},
    }
    config = _make_config(
        size=90,
        provided_labels=labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance=balance_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    result = create_splits(config, datasets)

    train_indices = result["train"]["indexes"].tolist()
    train_weights = result["train"]["weights"]
    assert train_weights is not None

    primary_ds = datasets["train"].prepped_datasets["data"]
    for i, idx in enumerate(train_indices):
        label = primary_ds.get_label(idx)
        if label == "C":
            assert train_weights[i] == 0.0, f"expected weight 0.0 for label C at dataset index {idx}"
        else:
            assert train_weights[i] > 0.0, (
                f"expected positive weight for label {label!r} at dataset index {idx}"
            )


# ---------------------------------------------------------------------------
# Test 13: split value range validation – (0.0, 1.0]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.001, 0.5, 0.75, 0.99, 1.0])
def test_validate_split_config_valid_fraction_passes(tmp_path, fraction):
    """validate_split_config accepts any split value in (0.0, 1.0]."""
    from hyrax.splitting_utils import validate_split_config

    config = _make_config(
        size=50,
        tmp_path=tmp_path,
        split={"train": fraction},
        groups=("train",),
    )
    datasets = _make_providers(config, ("train",))

    # Must not raise
    validate_split_config(config, datasets)


@pytest.mark.parametrize("fraction", [-1.0, -0.001, 1.001, 1.5, 2.0])
def test_validate_split_config_out_of_range_raises(tmp_path, fraction):
    """validate_split_config raises RuntimeError for split values outside (0.0, 1.0]."""
    from hyrax.splitting_utils import validate_split_config

    config = _make_config(
        size=50,
        tmp_path=tmp_path,
        split={"train": fraction},
        groups=("train",),
    )
    datasets = _make_providers(config, ("train",))

    with pytest.raises(RuntimeError, match="out of range"):
        validate_split_config(config, datasets)


def test_validate_split_config_multiple_valid_groups_passes(tmp_path):
    """validate_split_config accepts multiple groups when every value is in (0.0, 1.0]."""
    from hyrax.splitting_utils import validate_split_config

    shared_loc = str(tmp_path / "shared_data")
    config = _make_config(
        size=100,
        tmp_path=tmp_path,
        split={"train": 0.6, "validate": 0.2, "test": 0.2},
        data_location=shared_loc,
        groups=("train", "validate", "test"),
    )
    datasets = _make_providers(config, ("train", "validate", "test"))

    # Must not raise
    validate_split_config(config, datasets)


def test_validate_split_config_different_location_excluded_from_sum(tmp_path):
    """A group at a different data_location is not counted toward the shared-location sum.

    train and validate share shared_loc with fractions summing to exactly 1.0.
    test lives at a separate location with no explicit split (defaults to 1.0).
    If the locations were not isolated, the combined total would be 2.0 and raise;
    correct behaviour is to pass silently.
    """
    from hyrax.splitting_utils import validate_split_config

    shared_loc = str(tmp_path / "shared_data")
    other_loc = str(tmp_path / "other_data")

    config = _make_config(
        size=100,
        tmp_path=tmp_path,
        split={"train": 0.6, "validate": 0.4},
        data_location=shared_loc,
        groups=("train", "validate", "test"),
    )
    # Redirect the test group to a separate location
    config["data_request"]["test"]["data"]["data_location"] = other_loc

    datasets = _make_providers(config, ("train", "validate", "test"))

    # Must not raise: shared_loc sum is 1.0; other_loc sum is 1.0 (independent)
    validate_split_config(config, datasets)


def test_validate_split_config_infer_excluded_from_sum(tmp_path):
    """infer is excluded from the shared-location sum check.

    train uses 0.8 of shared_loc; infer also points at shared_loc with its
    default fraction of 1.0.  If infer were counted, the combined sum would
    be 1.8 and raise.  Because infer is always treated as independent, only
    train's 0.8 is summed and validate_split_config passes.
    """
    from hyrax.splitting_utils import validate_split_config

    shared_loc = str(tmp_path / "shared_data")
    config = _make_config(
        size=100,
        tmp_path=tmp_path,
        split={"train": 0.8},
        data_location=shared_loc,
        groups=("train", "infer"),
    )
    datasets = _make_providers(config, ("train", "infer"))

    # Must not raise
    validate_split_config(config, datasets)


# ===========================================================================
# Additional spec-coverage tests
#
# The tests below fill gaps between specs/splits_and_balancing_spec.md and the
# scenarios covered above.  Each test cites the relevant spec section.
# ===========================================================================


# ---------------------------------------------------------------------------
# Test 14: validate_split_config — all-floats-or-all-paths and shared parent
# (spec §4.1 "Mixing fractions and paths", §7.2, §15.7)
# ---------------------------------------------------------------------------


def test_validate_split_config_mixed_float_and_path_raises(tmp_path):
    """Mixing a path value and a float value across groups → RuntimeError.

    The mixed check fires before any file-existence check, so the path need
    not exist for this validation to trigger.
    """
    from hyrax.splitting_utils import validate_split_config

    shared_loc = str(tmp_path / "shared_data")
    config = _make_config(
        size=50,
        tmp_path=tmp_path,
        split={"train": str(tmp_path / "train_split.npz"), "validate": 0.4},
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))

    with pytest.raises(RuntimeError, match="all floats or all paths"):
        validate_split_config(config, datasets)


def test_validate_split_config_differing_parent_dirs_raises(tmp_path):
    """Path-based split values that live in different parent dirs → RuntimeError."""
    from hyrax.splitting_utils import validate_split_config

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    config = _make_config(
        size=50,
        tmp_path=tmp_path,
        split={
            "train": str(dir_a / "train_split.npz"),
            "validate": str(dir_b / "validate_split.npz"),
        },
        data_location=str(tmp_path / "shared_data"),
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))

    with pytest.raises(RuntimeError, match="common parent directory"):
        validate_split_config(config, datasets)


# ---------------------------------------------------------------------------
# Test 15: validate_balance_config — pre-scan rules
# (spec §4.2, §7.2)
# ---------------------------------------------------------------------------


def test_validate_balance_config_missing_getter_raises(tmp_path):
    """balance.field with no matching get_<field> on the primary dataset → raise."""
    from hyrax.splitting_utils import validate_balance_config

    config = _make_config(
        size=30,
        tmp_path=tmp_path,
        balance={"field": "bogus", "groups": [], "distribution": {}},
        groups=("train",),
    )
    datasets = _make_providers(config, ("train",))

    with pytest.raises(RuntimeError, match="get_bogus"):
        validate_balance_config(config, datasets)


def test_validate_balance_config_field_unset_with_groups_raises(tmp_path):
    """balance.groups/distribution set while balance.field is falsy → raise."""
    from hyrax.splitting_utils import validate_balance_config

    config = _make_config(
        size=30,
        tmp_path=tmp_path,
        balance={"field": False, "groups": ["train"], "distribution": {}},
        groups=("train",),
    )
    datasets = _make_providers(config, ("train",))

    with pytest.raises(RuntimeError, match="balance.field must be set"):
        validate_balance_config(config, datasets)


def test_validate_balance_config_extra_group_warns(tmp_path, caplog):
    """A balance.groups entry absent from data_request warns (not an error)."""
    from hyrax.splitting_utils import validate_balance_config

    config = _make_config(
        size=30,
        provided_labels=["A", "B"],
        tmp_path=tmp_path,
        balance={"field": "label", "groups": ["train", "ghost"], "distribution": {}},
        groups=("train",),
    )
    datasets = _make_providers(config, ("train",))

    with caplog.at_level(logging.WARNING, logger="hyrax.splitting_utils"):
        validate_balance_config(config, datasets)  # must not raise

    assert "ghost" in caplog.text
    assert "not in data_request" in caplog.text


# ---------------------------------------------------------------------------
# Test 16: validate_distribution_labels — post-scan warn / no-op
# (spec §7.2)
# ---------------------------------------------------------------------------


def test_validate_distribution_labels_missing_observed_label_warns(caplog):
    """An observed class absent from a non-empty distribution warns (weight 0),
    and does not raise."""
    from hyrax.splitting_utils import validate_distribution_labels

    with caplog.at_level(logging.WARNING, logger="hyrax.splitting_utils"):
        validate_distribution_labels({"A": 1.0}, {"A", "B"})

    assert "B" in caplog.text
    assert "absent from balance.distribution" in caplog.text


def test_validate_distribution_labels_empty_distribution_is_noop():
    """An empty distribution imposes no constraints on observed labels."""
    from hyrax.splitting_utils import validate_distribution_labels

    # Must not raise even though labels are present.
    validate_distribution_labels({}, {"A", "B"})


# ---------------------------------------------------------------------------
# Test 17: groups_to_balance table row 3 — distribution-only balances all
# non-infer groups (spec §4.2 table, §15.6)
# ---------------------------------------------------------------------------


def test_create_splits_distribution_only_balances_all_non_infer_groups(tmp_path):
    """balance.groups=[] + non-empty distribution → every non-infer group gets
    weights, while ``infer`` is always left unbalanced (the third row of the
    spec §4.2 table, which excludes ``infer``)."""
    from hyrax.splitting_utils import create_splits

    labels = ["A", "B"]
    shared_loc = str(tmp_path / "shared_data")
    config = _make_config(
        size=100,
        provided_labels=labels,
        tmp_path=tmp_path,
        split={"train": 0.6, "validate": 0.4, "rng_seed": 3},
        balance={"field": "label", "groups": [], "distribution": {"A": 0.5, "B": 0.5}},
        data_location=shared_loc,
        groups=("train", "validate", "infer"),
    )
    datasets = _make_providers(config, ("train", "validate", "infer"))
    result = create_splits(config, datasets)

    assert result["train"]["weights"] is not None, "train should be balanced"
    assert result["validate"]["weights"] is not None, "validate should be balanced too"
    assert result["infer"]["weights"] is None, "infer must never be balanced"


# ---------------------------------------------------------------------------
# Test 18: Weight values — uniform == inverse frequency, custom == raw
# target_c / count_c (spec §5, §15.5, §15.6)
# ---------------------------------------------------------------------------


def _split_label_counts(provider, indices):
    """Return (labels_per_index, count_per_label) for a split using get_label."""
    primary = provider.prepped_datasets["data"]
    labels = [str(primary.get_label(i)) for i in indices]
    counts: dict[str, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return labels, counts


def test_create_splits_uniform_weights_are_inverse_frequency(tmp_path):
    """Equal rebalance (empty distribution) → w_i = (1/K) / count_{class(i)}."""
    from hyrax.splitting_utils import create_splits

    labels = ["A", "B"]
    shared_loc = str(tmp_path / "shared_data")
    config = _make_config(
        size=120,
        provided_labels=labels,
        tmp_path=tmp_path,
        split={"train": 0.7, "validate": 0.3, "rng_seed": 4},
        balance={"field": "label", "groups": ["train"], "distribution": {}},
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    result = create_splits(config, datasets)

    train_idx = result["train"]["indexes"].tolist()
    weights = result["train"]["weights"]
    labels_for, counts = _split_label_counts(datasets["train"], train_idx)

    uniform = 1.0 / len(labels)  # K observed classes
    for w, lbl in zip(weights, labels_for):
        assert np.isclose(w, uniform / counts[lbl])


def test_create_splits_custom_distribution_weight_values_are_raw(tmp_path):
    """Custom distribution → w_i = target_{class(i)} / count_{class(i)} (raw,
    not normalised)."""
    from hyrax.splitting_utils import create_splits

    labels = ["A", "B"]
    shared_loc = str(tmp_path / "shared_data")
    dist = {"A": 0.6, "B": 0.4}
    config = _make_config(
        size=120,
        provided_labels=labels,
        tmp_path=tmp_path,
        split={"train": 0.8, "validate": 0.2, "rng_seed": 8},
        balance={"field": "label", "groups": ["train"], "distribution": dist},
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    result = create_splits(config, datasets)

    train_idx = result["train"]["indexes"].tolist()
    weights = result["train"]["weights"]
    labels_for, counts = _split_label_counts(datasets["train"], train_idx)

    for w, lbl in zip(weights, labels_for):
        assert np.isclose(w, dist[lbl] / counts[lbl])


# ---------------------------------------------------------------------------
# Test 19: infer compute branch — first-N contiguous, unshuffled, weights None
# (spec §4.1, §7.3)
# ---------------------------------------------------------------------------


def test_create_splits_infer_first_n_contiguous_no_shuffle(tmp_path):
    """The infer group takes the first round(N*frac) indices with no shuffle."""
    from hyrax.splitting_utils import create_splits

    size = 20
    config = _make_config(
        size=size,
        tmp_path=tmp_path,
        split={"infer": 0.5, "rng_seed": 1},
        groups=("infer",),
    )
    datasets = _make_providers(config, ("infer",))
    result = create_splits(config, datasets)

    assert result["infer"]["indexes"].tolist() == list(range(10))
    assert result["infer"]["weights"] is None


# ---------------------------------------------------------------------------
# Test 20: Σ < 1.0 on a shared location leaves a subset unused
# (spec §4.1, §15.2)
# ---------------------------------------------------------------------------


def test_create_splits_partial_fractions_leave_subset_unused(tmp_path):
    """train=0.5 + validate=0.3 (Σ=0.8) on a shared location assigns only 80% of
    the items; the rest are left unused and the splits do not overlap."""
    from hyrax.splitting_utils import create_splits

    size = 100
    shared_loc = str(tmp_path / "shared_data")
    config = _make_config(
        size=size,
        tmp_path=tmp_path,
        split={"train": 0.5, "validate": 0.3, "rng_seed": 9},
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    result = create_splits(config, datasets)

    train_idx = result["train"]["indexes"].tolist()
    val_idx = result["validate"]["indexes"].tolist()

    assert len(train_idx) == 50
    assert len(val_idx) == 30
    assert set(train_idx).isdisjoint(set(val_idx))
    assert len(set(train_idx) | set(val_idx)) == 80  # 20 items unused


# ---------------------------------------------------------------------------
# Test 21: distinct data_locations get independent fractions
# (spec §4.1)
# ---------------------------------------------------------------------------


def test_create_splits_distinct_locations_independent_fractions(tmp_path):
    """Two groups at different data_locations each take a fraction of their own
    source, so a 0.6 fraction on each is valid even though the sum exceeds 1.0."""
    from hyrax.splitting_utils import create_splits

    size = 100
    shared_loc = str(tmp_path / "shared_data")
    other_loc = str(tmp_path / "other_data")
    config = _make_config(
        size=size,
        tmp_path=tmp_path,
        split={"train": 0.6, "validate": 0.6, "rng_seed": 1},
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    # Redirect validate to its own source.
    config["data_request"]["validate"]["data"]["data_location"] = other_loc
    datasets = _make_providers(config, ("train", "validate"))

    result = create_splits(config, datasets)

    assert len(result["train"]["indexes"]) == 60
    assert len(result["validate"]["indexes"]) == 60


# ---------------------------------------------------------------------------
# Test 22: configs_equivalent — identical and per-field change detection
# (spec §7.5, §15.8)
# ---------------------------------------------------------------------------


def _equiv_base_config() -> dict:
    """Minimal config dict carrying every field configs_equivalent inspects."""
    return {
        "data_set": {"seed": 42},
        "split": {"train": 0.8, "validate": 0.2, "rng_seed": 5},
        "balance": {"field": "label", "groups": ["train"], "distribution": {"A": 0.5, "B": 0.5}},
        "data_request": {
            "train": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "/loc",
                    "primary_id_field": "object_id",
                }
            },
            "validate": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "/loc",
                    "primary_id_field": "object_id",
                }
            },
        },
    }


def test_configs_equivalent_identical_configs():
    """Two structurally identical configs are equivalent with no diffs."""
    from hyrax.splitting_utils import configs_equivalent

    base = _equiv_base_config()
    equivalent, diffs = configs_equivalent(base, copy.deepcopy(base))

    assert equivalent
    assert diffs == []


def _mut_field(c):
    c["balance"]["field"] = "other"


def _mut_distribution(c):
    c["balance"]["distribution"] = {"A": 0.7, "B": 0.3}


def _mut_rng_seed(c):
    c["split"]["rng_seed"] = 999


def _mut_dataset_class(c):
    c["data_request"]["train"]["data"]["dataset_class"] = "Other"


def _mut_data_location(c):
    c["data_request"]["train"]["data"]["data_location"] = "/other"


def _mut_fraction(c):
    c["split"]["train"] = 0.5


def _mut_groups(c):
    c["balance"]["groups"] = ["validate"]


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (_mut_field, "balance.field"),
        (_mut_distribution, "balance.distribution"),
        (_mut_rng_seed, "rng_seed"),
        (_mut_dataset_class, "dataset_class"),
        (_mut_data_location, "data_location"),
        (_mut_fraction, "split.train"),
        (_mut_groups, "balance.groups membership"),
    ],
)
def test_configs_equivalent_detects_changes(mutate, expected):
    """Each equivalency-relevant field, when changed, makes configs non-equivalent
    and is named in the diff list (spec §7.5)."""
    from hyrax.splitting_utils import configs_equivalent

    base = _equiv_base_config()
    cur = copy.deepcopy(base)
    mutate(cur)

    equivalent, diffs = configs_equivalent(base, cur)

    assert not equivalent
    assert any(expected in d for d in diffs), f"expected '{expected}' in diffs, got {diffs}"


# ---------------------------------------------------------------------------
# Test 23: path input with a differing sibling split_config.toml warns
# (spec §3 D10, §7.1 step 2)
# ---------------------------------------------------------------------------


def test_create_splits_path_sibling_config_difference_warns(tmp_path, caplog):
    """Supplying split files whose sibling split_config.toml differs from the
    current config logs a warning but still loads the files."""
    from hyrax.splitting_utils import create_splits, persist_splits

    size = 40
    config = _make_config(
        size=size,
        tmp_path=tmp_path,
        split={"train": 0.5, "rng_seed": 5},
        groups=("train",),
    )
    datasets = _make_providers(config, ("train",))
    result = create_splits(config, datasets)
    persist_splits(tmp_path, result, config)  # writes train_split.npz + split_config.toml

    train_npz = tmp_path / "train_split.npz"

    # Reload via path but with a different rng_seed → sibling config differs.
    config2 = _make_config(
        size=size,
        tmp_path=tmp_path,
        split={"train": str(train_npz), "rng_seed": 999},
        groups=("train",),
    )
    datasets2 = _make_providers(config2, ("train",))

    with caplog.at_level(logging.WARNING, logger="hyrax.splitting_utils"):
        create_splits(config2, datasets2)

    assert "different config" in caplog.text
    assert datasets2["train"].split_indices is not None


# ---------------------------------------------------------------------------
# Test 24: load_split_files error paths (spec §7.4)
# ---------------------------------------------------------------------------


def test_load_split_files_missing_indexes_array_raises(tmp_path):
    """An .npz without an 'indexes' array → RuntimeError."""
    from hyrax.splitting_utils import load_split_files

    bad_npz = tmp_path / "train_split.npz"
    np.savez_compressed(bad_npz, other=np.array([1, 2, 3]))

    with pytest.raises(RuntimeError, match="missing the required 'indexes'"):
        load_split_files({"train": bad_npz})


def test_load_split_files_missing_file_raises(tmp_path):
    """A path that does not exist → RuntimeError."""
    from hyrax.splitting_utils import load_split_files

    with pytest.raises(RuntimeError, match="not found"):
        load_split_files({"train": tmp_path / "does_not_exist.npz"})


def test_load_split_files_round_trips_weights(tmp_path):
    """A persisted balanced group reloads its weights as an ndarray."""
    from hyrax.splitting_utils import create_splits, load_split_files

    shared_loc = str(tmp_path / "shared_data")
    config = _make_config(
        size=80,
        provided_labels=["A", "B"],
        tmp_path=tmp_path,
        split={"train": 0.75, "validate": 0.25, "rng_seed": 6},
        balance={"field": "label", "groups": ["train"], "distribution": {}},
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    create_splits(config, datasets, results_dir=tmp_path, persist=True)

    loaded = load_split_files(
        {
            "train": tmp_path / "train_split.npz",
            "validate": tmp_path / "validate_split.npz",
        }
    )

    assert loaded["train"]["weights"] is not None
    assert isinstance(loaded["train"]["weights"], np.ndarray)
    assert loaded["validate"]["weights"] is None


# ---------------------------------------------------------------------------
# Test 25: rng_seed semantics (spec §4.1)
# ---------------------------------------------------------------------------


def test_create_splits_rng_seed_false_uses_data_set_seed(tmp_path):
    """rng_seed=false falls back to config['data_set']['seed']: identical seeds
    give identical splits; a different seed gives a different split."""
    from hyrax.splitting_utils import create_splits

    shared_loc = str(tmp_path / "shared_data")

    def build(ds_seed):
        cfg = _make_config(
            size=100,
            tmp_path=tmp_path,
            split={"train": 0.6, "validate": 0.4, "rng_seed": False},
            data_location=shared_loc,
            groups=("train", "validate"),
        )
        cfg["data_set"]["seed"] = ds_seed
        return cfg

    cfg_a = build(42)
    ds_a = _make_providers(cfg_a, ("train", "validate"))
    create_splits(cfg_a, ds_a)

    cfg_b = build(42)
    ds_b = _make_providers(cfg_b, ("train", "validate"))
    create_splits(cfg_b, ds_b)

    # Same fallback seed → identical splits.
    assert ds_a["train"].split_indices == ds_b["train"].split_indices

    cfg_c = build(7)
    ds_c = _make_providers(cfg_c, ("train", "validate"))
    create_splits(cfg_c, ds_c)

    # Different fallback seed → different shuffle.
    assert ds_a["train"].split_indices != ds_c["train"].split_indices


# ---------------------------------------------------------------------------
# Test 25a: _resolve_seed — exhaustive unit tests for the fix
# Regression for: config.get("data_set", {}).get("seed") instead of direct access
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rng_seed_value", [False, None, ""])
def test_resolve_seed_falsy_rng_seed_no_data_set_returns_none(rng_seed_value):
    """When rng_seed is falsy and data_set is absent entirely, return None (not KeyError)."""
    from hyrax.splitting_utils import _resolve_seed

    config = {"split": {"rng_seed": rng_seed_value}}
    assert _resolve_seed(config) is None


@pytest.mark.parametrize("rng_seed_value", [False, None, ""])
def test_resolve_seed_falsy_rng_seed_data_set_missing_seed_returns_none(rng_seed_value):
    """When rng_seed is falsy and data_set has no 'seed' key, return None."""
    from hyrax.splitting_utils import _resolve_seed

    config = {"split": {"rng_seed": rng_seed_value}, "data_set": {}}
    assert _resolve_seed(config) is None


@pytest.mark.parametrize("rng_seed_value", [False, None, ""])
def test_resolve_seed_falsy_rng_seed_uses_data_set_seed(rng_seed_value):
    """When rng_seed is falsy and data_set.seed is set, return data_set.seed."""
    from hyrax.splitting_utils import _resolve_seed

    config = {"split": {"rng_seed": rng_seed_value}, "data_set": {"seed": 99}}
    assert _resolve_seed(config) == 99


def test_resolve_seed_truthy_rng_seed_ignores_data_set():
    """When rng_seed is a truthy integer, return it regardless of data_set."""
    from hyrax.splitting_utils import _resolve_seed

    config = {"split": {"rng_seed": 7}, "data_set": {"seed": 999}}
    assert _resolve_seed(config) == 7


def test_resolve_seed_truthy_rng_seed_no_data_set():
    """When rng_seed is set, data_set being absent does not matter."""
    from hyrax.splitting_utils import _resolve_seed

    config = {"split": {"rng_seed": 42}}
    assert _resolve_seed(config) == 42


def test_configs_equivalent_no_data_set_key_does_not_raise():
    """configs_equivalent must not crash when either config lacks a data_set key."""
    from hyrax.splitting_utils import configs_equivalent

    base = {
        "split": {"rng_seed": 5, "train": 0.8},
        "balance": {"field": False, "groups": [], "distribution": {}},
        "data_request": {
            "train": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "/loc",
                    "primary_id_field": "object_id",
                }
            }
        },
    }
    other = {**base, "split": {**base["split"], "rng_seed": 5}}

    equivalent, diffs = configs_equivalent(base, other)
    assert equivalent
    assert diffs == []


def test_create_splits_string_rng_seed_raises(tmp_path):
    """A non-empty string rng_seed is rejected (must be an integer or false)."""
    from hyrax.splitting_utils import create_splits

    config = _make_config(
        size=20,
        tmp_path=tmp_path,
        split={"train": 1.0, "rng_seed": "abc"},
        groups=("train",),
    )
    datasets = _make_providers(config, ("train",))

    with pytest.raises(RuntimeError, match="must be an integer"):
        create_splits(config, datasets)


# ---------------------------------------------------------------------------
# Test 26: DataProvider.__repr__ reflects split selection and rebalancing
# (spec §8)
# ---------------------------------------------------------------------------


def test_data_provider_repr_reflects_split_and_rebalance(tmp_path):
    """After create_splits, __repr__ shows the selected-item count and marks a
    rebalanced (weighted) group."""
    from hyrax.splitting_utils import create_splits

    shared_loc = str(tmp_path / "shared_data")
    config = _make_config(
        size=80,
        provided_labels=["A", "B"],
        tmp_path=tmp_path,
        split={"train": 0.7, "validate": 0.3, "rng_seed": 2},
        balance={"field": "label", "groups": ["train"], "distribution": {}},
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets = _make_providers(config, ("train", "validate"))
    create_splits(config, datasets)

    train_repr = repr(datasets["train"])
    val_repr = repr(datasets["validate"])

    assert "Selected items:" in train_repr
    assert "(rebalanced)" in train_repr  # train carries weights
    assert "Selected items:" in val_repr
    assert "(rebalanced)" not in val_repr  # validate is unweighted
