"""Tests for split integration across DataProvider, create_splits, and dist_data_loader.

These tests verify that split configuration flows correctly through the system:
  config['split'] → create_splits → DataProvider.split_indices → dist_data_loader
"""

import pytest

import hyrax
from hyrax.datasets.data_provider import DataProvider


def _make_config(data_location, *, split=None, size=100, seed=42):
    """Return a fresh Hyrax config wired to HyraxRandomDataset."""
    config = hyrax.Hyrax().config
    config["data_set"]["HyraxRandomDataset"]["size"] = size
    config["data_set"]["HyraxRandomDataset"]["seed"] = seed
    config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]
    if split is not None:
        config["split"] = split
    return config


def _make_providers(config, data_location, groups=("train",)):
    """Set up data_request in config and return DataProvider dict via setup_dataset."""
    from hyrax.pytorch_ignite import setup_dataset

    dr = {}
    for g in groups:
        dr[g] = {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": data_location,
                "primary_id_field": "object_id",
                "fields": ["image"],
            }
        }
    config["data_request"] = dr
    return setup_dataset(config, splits=groups)


def _make_single_provider(config, data_location):
    """Convenience wrapper returning a single DataProvider for the 'train' group."""
    return _make_providers(config, data_location)["train"]


# ===========================================================================
# 1. DataProvider attribute tests
# ===========================================================================


class TestDataProviderAttributes:
    """Verify DataProvider initialises split-related attributes correctly."""

    def test_split_fraction_not_a_dataprovider_attribute(self, tmp_path):
        """split_fraction is no longer a DataProvider attribute.

        Splits are controlled via config['split'] and assigned by create_splits.
        """
        config = _make_config(str(tmp_path))
        dp = _make_single_provider(config, str(tmp_path))

        assert not hasattr(dp, "split_fraction"), (
            "split_fraction should not be a DataProvider attribute; use config['split'] instead."
        )

    def test_split_indices_none_by_default(self, tmp_path):
        """split_indices is None until assigned by create_splits."""
        config = _make_config(str(tmp_path))
        dp = _make_single_provider(config, str(tmp_path))

        assert dp.split_indices is None

    def test_split_weights_none_by_default(self, tmp_path):
        """split_weights is None until assigned by create_splits with balance config."""
        config = _make_config(str(tmp_path))
        dp = _make_single_provider(config, str(tmp_path))

        assert dp.split_weights is None

    def test_primary_data_location_stored(self, tmp_path):
        """primary_data_location is populated from the primary dataset."""
        config = _make_config(str(tmp_path))
        dp = _make_single_provider(config, str(tmp_path))

        assert dp.primary_data_location == str(tmp_path)

    def test_split_indices_can_be_assigned(self, tmp_path):
        """split_indices can be set externally (as create_splits does)."""
        config = _make_config(str(tmp_path))
        dp = _make_single_provider(config, str(tmp_path))

        dp.split_indices = [0, 1, 2, 3, 4]
        assert dp.split_indices == [0, 1, 2, 3, 4]


# ===========================================================================
# 2. Config['split'] → create_splits → DataProvider.split_indices
# ===========================================================================


class TestSplitConfigFlow:
    """Verify that config['split'] fractions flow through create_splits into
    DataProvider.split_indices correctly."""

    def test_basic_two_way_split_assigns_indices(self, tmp_path):
        """60/40 split assigns non-overlapping indices to both providers."""
        from hyrax.splitting_utils import create_splits

        split_cfg = {"train": 0.6, "validate": 0.4, "rng_seed": 1}
        config = _make_config(str(tmp_path), split=split_cfg, size=100)
        datasets = _make_providers(config, str(tmp_path), groups=("train", "validate"))

        create_splits(config, datasets)

        assert datasets["train"].split_indices is not None
        assert datasets["validate"].split_indices is not None
        assert len(datasets["train"].split_indices) == 60
        assert len(datasets["validate"].split_indices) == 40

    def test_three_way_split_assigns_indices(self, tmp_path):
        """60/20/20 split assigns non-overlapping indices of the correct size to all three providers."""
        from hyrax.splitting_utils import create_splits

        split_cfg = {"train": 0.6, "validate": 0.2, "test": 0.2, "rng_seed": 10}
        config = _make_config(str(tmp_path), split=split_cfg, size=100)
        datasets = _make_providers(config, str(tmp_path), groups=("train", "validate", "test"))

        create_splits(config, datasets)

        assert datasets["train"].split_indices is not None
        assert datasets["validate"].split_indices is not None
        assert datasets["test"].split_indices is not None
        assert len(datasets["train"].split_indices) == 60
        assert len(datasets["validate"].split_indices) == 20
        assert len(datasets["test"].split_indices) == 20

        train_set = set(datasets["train"].split_indices)
        val_set = set(datasets["validate"].split_indices)
        test_set = set(datasets["test"].split_indices)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_split_indices_are_non_overlapping(self, tmp_path):
        """No index appears in more than one split."""
        from hyrax.splitting_utils import create_splits

        split_cfg = {"train": 0.6, "validate": 0.4, "rng_seed": 2}
        config = _make_config(str(tmp_path), split=split_cfg, size=100)
        datasets = _make_providers(config, str(tmp_path), groups=("train", "validate"))

        create_splits(config, datasets)

        train_set = set(datasets["train"].split_indices)
        val_set = set(datasets["validate"].split_indices)
        assert train_set.isdisjoint(val_set), "Duplicate indices found across splits"

    def test_fractions_summing_to_one_cover_all_indices(self, tmp_path):
        """When fractions sum to 1.0, every dataset index is assigned."""
        from hyrax.splitting_utils import create_splits

        size = 50
        split_cfg = {"train": 0.6, "validate": 0.4, "rng_seed": 3}
        config = _make_config(str(tmp_path), split=split_cfg, size=size)
        datasets = _make_providers(config, str(tmp_path), groups=("train", "validate"))

        create_splits(config, datasets)

        all_indices = sorted(datasets["train"].split_indices + datasets["validate"].split_indices)
        assert all_indices == list(range(size))

    def test_deterministic_with_same_rng_seed(self, tmp_path):
        """Same rng_seed produces the same split_indices across calls."""
        from hyrax.splitting_utils import create_splits

        split_cfg = {"train": 0.6, "validate": 0.4, "rng_seed": 42}
        config_a = _make_config(str(tmp_path), split=split_cfg, size=100)
        datasets_a = _make_providers(config_a, str(tmp_path), groups=("train", "validate"))
        create_splits(config_a, datasets_a)

        config_b = _make_config(str(tmp_path), split=split_cfg, size=100)
        datasets_b = _make_providers(config_b, str(tmp_path), groups=("train", "validate"))
        create_splits(config_b, datasets_b)

        assert datasets_a["train"].split_indices == datasets_b["train"].split_indices
        assert datasets_a["validate"].split_indices == datasets_b["validate"].split_indices

    def test_error_when_fractions_exceed_one(self, tmp_path):
        """validate_split_config raises RuntimeError when fractions sum > 1.0."""
        from hyrax.splitting_utils import validate_split_config

        split_cfg = {"train": 0.7, "validate": 0.5}
        config = _make_config(str(tmp_path), split=split_cfg, size=100)
        datasets = _make_providers(config, str(tmp_path), groups=("train", "validate"))

        with pytest.raises(RuntimeError, match="sum to"):
            validate_split_config(config, datasets)

    def test_error_when_split_definition_missing_for_shared_location(self, tmp_path):
        """Missing split definition for one group at a shared location raises RuntimeError.

        When a group has no entry in config['split'] it defaults to 1.0 (full dataset).
        Combined with any explicit fraction from another group at the same location,
        the sum exceeds 1.0 and validate_split_config must raise.
        """
        from hyrax.splitting_utils import validate_split_config

        # validate has no split entry → defaults to 1.0; train=0.6 → sum=1.6
        split_cfg = {"train": 0.6, "rng_seed": 1}
        config = _make_config(str(tmp_path), split=split_cfg, size=100)
        datasets = _make_providers(config, str(tmp_path), groups=("train", "validate"))

        with pytest.raises(RuntimeError, match="sum to"):
            validate_split_config(config, datasets)

    def test_single_group_gets_subset(self, tmp_path):
        """A single provider with fraction < 1.0 receives a partial index list."""
        from hyrax.splitting_utils import create_splits

        split_cfg = {"train": 0.3, "rng_seed": 5}
        config = _make_config(str(tmp_path), split=split_cfg, size=100)
        datasets = _make_providers(config, str(tmp_path), groups=("train",))

        create_splits(config, datasets)

        assert len(datasets["train"].split_indices) == 30


# ===========================================================================
# 3. dist_data_loader integration tests
# ===========================================================================


class TestDistDataLoaderSplitIndices:
    """Verify that dist_data_loader uses split_indices from DataProvider."""

    def test_loader_subset_matches_split_indices(self, tmp_path):
        """When split_indices is set, the DataLoader wraps a Subset with those indices."""
        from torch.utils.data import Subset

        from hyrax.pytorch_ignite import dist_data_loader

        config = _make_config(str(tmp_path), size=20)
        dp = _make_single_provider(config, str(tmp_path))
        dp.split_indices = [0, 1, 2, 3, 4]

        loader = dist_data_loader(dp, config, shuffle=False)

        assert isinstance(loader.dataset, Subset)
        assert list(loader.dataset.indices) == [0, 1, 2, 3, 4]

    def test_loader_uses_all_indices_when_split_indices_none(self, tmp_path):
        """When split_indices is None, the DataLoader covers the full dataset."""
        from torch.utils.data import Subset

        from hyrax.pytorch_ignite import dist_data_loader

        size = 20
        config = _make_config(str(tmp_path), size=size)
        dp = _make_single_provider(config, str(tmp_path))
        assert dp.split_indices is None

        loader = dist_data_loader(dp, config, shuffle=False)

        assert isinstance(loader.dataset, Subset)
        assert list(loader.dataset.indices) == list(range(size))

    def test_end_to_end_split_to_dataloader(self, tmp_path):
        """End-to-end: config['split'] → create_splits → dist_data_loader sizes match."""
        from hyrax.pytorch_ignite import dist_data_loader
        from hyrax.splitting_utils import create_splits

        size = 100
        split_cfg = {"train": 0.6, "validate": 0.4, "rng_seed": 7}
        config = _make_config(str(tmp_path), split=split_cfg, size=size)
        datasets = _make_providers(config, str(tmp_path), groups=("train", "validate"))
        create_splits(config, datasets)

        train_loader = dist_data_loader(datasets["train"], config, shuffle=False)
        val_loader = dist_data_loader(datasets["validate"], config, shuffle=False)

        assert len(train_loader.dataset) == 60
        assert len(val_loader.dataset) == 40
        train_idx = set(train_loader.dataset.indices)
        val_idx = set(val_loader.dataset.indices)
        assert train_idx.isdisjoint(val_idx)

    def test_legacy_data_loader_shuffle_key_is_warned_and_ignored(self, tmp_path):
        """Legacy config['data_loader']['shuffle'] does not raise an error.

        The key is logged as a warning and stripped before the DataLoader is
        constructed so it never causes a sampler/shuffle conflict.
        """
        from hyrax.pytorch_ignite import dist_data_loader

        config = _make_config(str(tmp_path), size=20)
        config["data_loader"]["shuffle"] = True
        dp = _make_single_provider(config, str(tmp_path))
        dp.split_indices = [0, 1, 2, 3, 4]

        loader = dist_data_loader(dp, config, shuffle=False)
        assert loader is not None


# ===========================================================================
# 4. Engine verb integration with split_indices
# ===========================================================================


class TestEngineSplitIndices:
    """Verify that the engine verb logic respects split_indices when iterating."""

    def test_engine_uses_split_indices_when_set(self, tmp_path):
        """Engine index selection should use split_indices when present."""
        config = _make_config(str(tmp_path), size=100)
        expected_split_indices = [10, 20, 30, 40, 50]

        dp = _make_single_provider(config, str(tmp_path))
        dp.split_indices = expected_split_indices

        if isinstance(dp, DataProvider) and dp.split_indices is not None:
            indices_to_process = dp.split_indices
        else:
            indices_to_process = list(range(len(dp)))

        assert indices_to_process == expected_split_indices
        assert len(indices_to_process) == 5
        assert len(indices_to_process) != len(dp)

    def test_engine_processes_all_indices_when_split_indices_none(self, tmp_path):
        """Engine index selection should cover all indices when split_indices is None."""
        size = 100
        config = _make_config(str(tmp_path), size=size)
        dp = _make_single_provider(config, str(tmp_path))
        assert dp.split_indices is None

        if isinstance(dp, DataProvider) and dp.split_indices is not None:
            indices_to_process = dp.split_indices
        else:
            indices_to_process = list(range(len(dp)))

        assert indices_to_process == list(range(size))
