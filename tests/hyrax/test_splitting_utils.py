"""Tests for splitting_utils: create_splits, validate_split_config, persist/load."""

from __future__ import annotations

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
# Test 9: dist_data_loader Subset + sampler types
# ---------------------------------------------------------------------------


def test_dist_data_loader_sampler_types(tmp_path):
    """dist_data_loader creates a Subset restricted to split_indices.

    - WeightedRandomSampler when split_weights is set
    - SubsetRandomSampler when shuffle=True and no weights
    - No explicit sampler (sequential) when shuffle=False and no weights
    """
    from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler

    from hyrax.pytorch_ignite import dist_data_loader
    from hyrax.splitting_utils import create_splits

    size = 60
    shared_loc = str(tmp_path / "shared_data")
    labels = ["A", "B"]

    # ── weighted sampler ──────────────────────────────────────────────────
    split_cfg = {"train": 0.7, "validate": 0.3, "rng_seed": 11}
    balance_cfg = {"field": "label", "groups": ["train"], "distribution": {}}
    config_w = _make_config(
        size=size,
        provided_labels=labels,
        tmp_path=tmp_path,
        split=split_cfg,
        balance=balance_cfg,
        data_location=shared_loc,
        groups=("train", "validate"),
    )
    datasets_w = _make_providers(config_w, ("train", "validate"))
    create_splits(config_w, datasets_w)

    loader_weighted = dist_data_loader(datasets_w["train"], config_w, shuffle=True)
    assert isinstance(loader_weighted.sampler, WeightedRandomSampler), (
        "Expected WeightedRandomSampler when split_weights is set"
    )

    # ── SubsetRandomSampler (shuffle, no weights) ─────────────────────────
    config_s = _make_config(
        size=size,
        tmp_path=tmp_path,
        split={"train": 0.8, "rng_seed": 12},
        data_location=shared_loc,
        groups=("train",),
    )
    datasets_s = _make_providers(config_s, ("train",))
    create_splits(config_s, datasets_s)

    loader_shuffle = dist_data_loader(datasets_s["train"], config_s, shuffle=True)
    assert isinstance(loader_shuffle.sampler, SubsetRandomSampler), (
        "Expected SubsetRandomSampler when shuffle=True and no weights"
    )

    # ── sequential (no sampler) ───────────────────────────────────────────
    config_seq = _make_config(
        size=size,
        tmp_path=tmp_path,
        split={"train": 0.8, "rng_seed": 13},
        data_location=shared_loc,
        groups=("train",),
    )
    datasets_seq = _make_providers(config_seq, ("train",))
    create_splits(config_seq, datasets_seq)

    loader_seq = dist_data_loader(datasets_seq["train"], config_seq, shuffle=False)
    # No sampler → DataLoader uses its own default SequentialSampler over the Subset
    assert not isinstance(loader_seq.sampler, (WeightedRandomSampler, SubsetRandomSampler)), (
        "Expected no weighted/random sampler when shuffle=False and no weights"
    )

    # Verify all loaders reference a Subset dataset
    from torch.utils.data import Subset

    for loader in (loader_weighted, loader_shuffle, loader_seq):
        assert isinstance(loader.dataset, Subset), "DataLoader should wrap a Subset"


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
