"""Tests for the split_fraction integration across DataProvider,
create_splits_from_fractions, and dist_data_loader.

These tests verify that split_fraction values defined in data_request
configs flow correctly through the system:
  DataRequestConfig → DataProvider → create_splits_from_fractions → dist_data_loader
"""

import pytest

import hyrax
from hyrax.data_sets.data_provider import DataProvider
from hyrax.pytorch_ignite import create_splits_from_fractions


def _make_config():
    """Return a fresh Hyrax config dict."""
    return hyrax.Hyrax().config


def _make_provider(config, data_location, split_fraction=None, size=100):
    """Create a real DataProvider with a HyraxRandomDataset.

    Note: This mutates ``config`` in-place to set dataset size/seed/shape.
    Callers should use a fresh config (via ``_make_config()``) per test or
    per provider group to avoid cross-contamination.
    """
    request = {
        "data": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": data_location,
            "primary_id_field": "object_id",
            "fields": ["image"],
        }
    }
    if split_fraction is not None:
        request["data"]["split_fraction"] = split_fraction

    config["data_set"]["HyraxRandomDataset"]["size"] = size
    config["data_set"]["HyraxRandomDataset"]["seed"] = 42
    config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    dp = DataProvider(config, request)
    return dp


def _make_stub_provider(length, split_fraction=None):
    """Create a minimal stub that quacks like a DataProvider for
    create_splits_from_fractions (has __len__ and split_fraction)."""

    class _StubProvider:
        def __init__(self, n, frac):
            self.split_fraction = frac
            self._length = n

        def __len__(self):
            return self._length

    return _StubProvider(length, split_fraction)


# ===========================================================================
# 1. DataProvider attribute tests
# ===========================================================================


class TestDataProviderSplitFraction:
    """Verify that DataProvider correctly extracts split_fraction,
    primary_data_location, and initialises split_indices."""

    def test_split_fraction_extracted_from_primary_dataset(self):
        """split_fraction is read from the primary dataset definition."""
        config = _make_config()
        dp = _make_provider(config, "./data/test", split_fraction=0.6)

        assert dp.split_fraction == 0.6

    def test_split_fraction_none_when_not_set(self):
        """split_fraction defaults to None when not in the data request."""
        config = _make_config()
        dp = _make_provider(config, "./data/test", split_fraction=None)

        assert dp.split_fraction is None

    def test_primary_data_location_stored(self):
        """primary_data_location is populated from the primary dataset."""
        config = _make_config()
        dp = _make_provider(config, "./data/my_location", split_fraction=0.5)

        # data_location is resolved to an absolute path by the Pydantic validator
        # when going through DataRequestConfig, but DataProvider reads from the
        # raw dict so it stores whatever was passed.
        assert dp.primary_data_location == "./data/my_location"

    def test_split_indices_none_by_default(self):
        """split_indices is None until set externally by setup_dataset."""
        config = _make_config()
        dp = _make_provider(config, "./data/test", split_fraction=0.7)

        assert dp.split_indices is None

    def test_split_indices_can_be_assigned(self):
        """split_indices can be set externally (by setup_dataset)."""
        config = _make_config()
        dp = _make_provider(config, "./data/test", split_fraction=0.5)

        dp.split_indices = [0, 1, 2, 3, 4]
        assert dp.split_indices == [0, 1, 2, 3, 4]


# ===========================================================================
# 2. create_splits_from_fractions tests
# ===========================================================================


class TestCreateSplitsFromFractions:
    """Verify that create_splits_from_fractions correctly partitions indices."""

    @pytest.fixture()
    def base_config(self):
        """Base config fixture with a fixed seed for deterministic shuffling."""
        config = _make_config()
        config["data_set"]["seed"] = 42
        return config

    def test_basic_two_way_split(self, base_config):
        """60/40 split produces correct count of indices."""
        providers = {
            "train": _make_stub_provider(100, split_fraction=0.6),
            "validate": _make_stub_provider(100, split_fraction=0.4),
        }

        result = create_splits_from_fractions(providers, base_config)

        assert set(result.keys()) == {"train", "validate"}
        assert len(result["train"]) == 60
        assert len(result["validate"]) == 40

    def test_three_way_split(self, base_config):
        """60/20/20 split across three groups."""
        providers = {
            "train": _make_stub_provider(100, split_fraction=0.6),
            "validate": _make_stub_provider(100, split_fraction=0.2),
            "test": _make_stub_provider(100, split_fraction=0.2),
        }

        result = create_splits_from_fractions(providers, base_config)

        assert len(result["train"]) == 60
        assert len(result["validate"]) == 20
        assert len(result["test"]) == 20

    def test_indices_are_non_overlapping(self, base_config):
        """No index appears in more than one split."""
        providers = {
            "train": _make_stub_provider(100, split_fraction=0.6),
            "validate": _make_stub_provider(100, split_fraction=0.4),
        }

        result = create_splits_from_fractions(providers, base_config)

        all_indices = result["train"] + result["validate"]
        assert len(set(all_indices)) == len(all_indices), "Duplicate indices found across splits"

    def test_indices_cover_full_range_when_fractions_sum_to_one(self, base_config):
        """When fractions sum to 1.0, every index is assigned."""
        providers = {
            "train": _make_stub_provider(50, split_fraction=0.6),
            "validate": _make_stub_provider(50, split_fraction=0.4),
        }

        result = create_splits_from_fractions(providers, base_config)

        all_indices = sorted(result["train"] + result["validate"])
        assert all_indices == list(range(50))

    def test_fractions_less_than_one_leave_unassigned(self, base_config):
        """When fractions sum to < 1.0, some indices are not assigned."""
        providers = {
            "train": _make_stub_provider(100, split_fraction=0.5),
            "validate": _make_stub_provider(100, split_fraction=0.2),
        }

        result = create_splits_from_fractions(providers, base_config)

        total_assigned = len(result["train"]) + len(result["validate"])
        assert total_assigned == 70
        assert total_assigned < 100

    def test_deterministic_with_seed(self, base_config):
        """Same seed produces the same split."""
        providers_a = {
            "train": _make_stub_provider(100, split_fraction=0.6),
            "validate": _make_stub_provider(100, split_fraction=0.4),
        }
        providers_b = {
            "train": _make_stub_provider(100, split_fraction=0.6),
            "validate": _make_stub_provider(100, split_fraction=0.4),
        }

        result_a = create_splits_from_fractions(providers_a, base_config)
        result_b = create_splits_from_fractions(providers_b, base_config)

        assert result_a["train"] == result_b["train"]
        assert result_a["validate"] == result_b["validate"]

    def test_different_seed_produces_different_split(self):
        """Different seeds produce different index assignments."""
        config_a = _make_config()
        config_a["data_set"]["seed"] = 1
        config_b = _make_config()
        config_b["data_set"]["seed"] = 2

        providers_a = {
            "train": _make_stub_provider(100, split_fraction=0.6),
            "validate": _make_stub_provider(100, split_fraction=0.4),
        }
        providers_b = {
            "train": _make_stub_provider(100, split_fraction=0.6),
            "validate": _make_stub_provider(100, split_fraction=0.4),
        }

        result_a = create_splits_from_fractions(providers_a, config_a)
        result_b = create_splits_from_fractions(providers_b, config_b)

        # With different seeds the shuffled order should differ
        assert result_a["train"] != result_b["train"]

    def test_error_when_fractions_exceed_one(self, base_config):
        """RuntimeError raised when fractions sum > 1.0."""
        providers = {
            "train": _make_stub_provider(100, split_fraction=0.7),
            "validate": _make_stub_provider(100, split_fraction=0.5),
        }

        with pytest.raises(RuntimeError, match="exceeds 1.0"):
            create_splits_from_fractions(providers, base_config)

    def test_error_when_split_fraction_missing(self, base_config):
        """RuntimeError raised when a provider has no split_fraction."""
        providers = {
            "train": _make_stub_provider(100, split_fraction=0.6),
            "validate": _make_stub_provider(100, split_fraction=None),
        }

        with pytest.raises(RuntimeError, match="does not have a split_fraction"):
            create_splits_from_fractions(providers, base_config)

    def test_single_provider(self, base_config):
        """A single provider with split_fraction < 1.0 gets a subset."""
        providers = {
            "train": _make_stub_provider(100, split_fraction=0.3),
        }

        result = create_splits_from_fractions(providers, base_config)

        assert len(result["train"]) == 30

    def test_small_dataset(self, base_config):
        """Rounding works correctly for small datasets."""
        providers = {
            "train": _make_stub_provider(7, split_fraction=0.6),
            "validate": _make_stub_provider(7, split_fraction=0.4),
        }

        result = create_splits_from_fractions(providers, base_config)

        # 7 * 0.6 = 4.2 → round to 4, 7 * 0.4 = 2.8 → round to 3
        total = len(result["train"]) + len(result["validate"])
        assert total == 7
        all_indices = sorted(result["train"] + result["validate"])
        assert all_indices == list(range(7))

    def test_clamping_prevents_overrun(self, base_config):
        """When rounding would overrun the index list, clamping kicks in."""
        # 3 * 0.5 = 1.5 → rounds to 2 for each, but 2+2=4 > 3
        providers = {
            "a": _make_stub_provider(3, split_fraction=0.5),
            "b": _make_stub_provider(3, split_fraction=0.5),
        }

        result = create_splits_from_fractions(providers, base_config)

        total = len(result["a"]) + len(result["b"])
        assert total <= 3, "Clamping should prevent exceeding the dataset size"
        # Indices should still be valid
        all_indices = result["a"] + result["b"]
        assert all(0 <= i < 3 for i in all_indices)

    def test_no_shuffle_preserves_order(self, base_config):
        """When shuffle=False, indices are assigned in natural order 0..N-1."""
        providers = {
            "train": _make_stub_provider(100, split_fraction=0.6),
            "validate": _make_stub_provider(100, split_fraction=0.4),
        }

        result = create_splits_from_fractions(providers, base_config, shuffle=False)

        # Without shuffling the first 60 indices should be 0-59 in order
        assert result["train"] == list(range(60))
        assert result["validate"] == list(range(60, 100))

    def test_shuffle_true_reorders_indices(self, base_config):
        """When shuffle=True (default), indices are NOT in natural order."""
        providers = {
            "train": _make_stub_provider(100, split_fraction=0.6),
            "validate": _make_stub_provider(100, split_fraction=0.4),
        }

        result = create_splits_from_fractions(providers, base_config, shuffle=True)

        # With shuffle, at least one split should differ from natural order
        assert result["train"] != list(range(60)) or result["validate"] != list(range(60, 100))


# ===========================================================================
# 3. dist_data_loader integration tests
# ===========================================================================


class TestDistDataLoaderSplitIndices:
    """Verify that dist_data_loader respects split_indices on DataProvider."""

    def test_uses_split_indices_when_set(self):
        """When split_indices is set, the dataloader only yields those indices."""
        from hyrax.pytorch_ignite import dist_data_loader

        config = _make_config()

        dp = _make_provider(config, "./data/test", size=20)

        # Manually set split_indices to a subset
        dp.split_indices = [0, 1, 2, 3, 4]

        _, returned_indices = dist_data_loader(dp, config, False)

        assert returned_indices == [0, 1, 2, 3, 4]

    def test_uses_all_indices_when_split_indices_not_set(self):
        """When split_indices is None, all indices are used."""
        from hyrax.pytorch_ignite import dist_data_loader

        config = _make_config()
        size = 20

        dp = _make_provider(config, "./data/test", size=size)
        assert dp.split_indices is None

        _, returned_indices = dist_data_loader(dp, config, False)

        assert returned_indices == list(range(size))

    def test_split_indices_count_matches_fraction(self):
        """End-to-end: create providers with fractions, compute splits,
        assign indices, and verify dist_data_loader returns correct counts."""
        from hyrax.pytorch_ignite import dist_data_loader

        config = _make_config()
        size = 100
        config["data_set"]["seed"] = 42

        train_dp = _make_provider(config, "./data/test", split_fraction=0.6, size=size)
        validate_dp = _make_provider(config, "./data/test", split_fraction=0.4, size=size)

        # Simulate what setup_dataset does
        providers = {"train": train_dp, "validate": validate_dp}
        split_indices = create_splits_from_fractions(providers, config)
        train_dp.split_indices = split_indices["train"]
        validate_dp.split_indices = split_indices["validate"]

        _, train_indices = dist_data_loader(train_dp, config, False)
        _, validate_indices = dist_data_loader(validate_dp, config, False)

        assert len(train_indices) == 60
        assert len(validate_indices) == 40
        assert set(train_indices).isdisjoint(set(validate_indices))
