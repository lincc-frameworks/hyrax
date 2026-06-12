# ruff: noqa: D102
"""Tests for the augmentation feature."""

import numpy as np
import pytest

import hyrax
from hyrax.datasets.data_provider import DataProvider
from hyrax.datasets.dataset_registry import HyraxDataset
from hyrax.datasets.random.hyrax_random_dataset import HyraxRandomDataset

# ---------------------------------------------------------------------------
# Test dataset classes (auto-registered in DATASET_REGISTRY on import)
# ---------------------------------------------------------------------------


class MinimalHyraxDataset(HyraxDataset):
    """Minimal HyraxDataset subclass for testing base-class on_epoch_start."""

    def __len__(self):
        return 0


class AugmentedRandomDataset(HyraxRandomDataset):
    """HyraxRandomDataset with augment_image for testing augmentation dispatch."""

    def __init__(self, config, data_location):
        super().__init__(config, data_location)
        self.image_getter_call_count = 0
        self.augment_image_calls: list[tuple[int, int]] = []
        self.epoch_start_count = 0
        self.last_verb: str | None = None

    def on_epoch_start(self, verb):
        self.epoch_start_count += 1
        self.last_verb = verb

    def get_image(self, idx):
        self.image_getter_call_count += 1
        return super().get_image(idx)

    def augment_image(self, data, idx, rng_seed):
        self.augment_image_calls.append((idx, rng_seed))
        return -data


class SeedTrackingDataset(HyraxRandomDataset):
    """HyraxRandomDataset that records rng_seeds for image and label augmentation."""

    def __init__(self, config, data_location):
        super().__init__(config, data_location)
        self.seeds_by_idx: dict[int, dict[str, int]] = {}

    def augment_image(self, data, idx, rng_seed):
        self.seeds_by_idx.setdefault(idx, {})["image"] = rng_seed
        return data

    def augment_label(self, data, idx, rng_seed):
        self.seeds_by_idx.setdefault(idx, {})["label"] = rng_seed
        return data


class CachingAugmentDataset(HyraxRandomDataset):
    """Dataset that caches augmented results by including rng_seed in cache key."""

    def augment_image(self, data, idx, rng_seed):
        return -data

    def row_cache_key(self, idx, rng_seed=None):
        if rng_seed is None:
            return np.int64(idx)
        return np.int64(idx * 1_000_000 + (rng_seed % 1_000_000))


class NoCacheDataset(HyraxRandomDataset):
    """Dataset whose row_cache_key always returns None (never caches)."""

    def __init__(self, config, data_location):
        super().__init__(config, data_location)
        self.get_image_call_count = 0

    def get_image(self, idx):
        self.get_image_call_count += 1
        return super().get_image(idx)

    def row_cache_key(self, idx, rng_seed=None):
        return None


class EpochCountingDataset(HyraxRandomDataset):
    """HyraxRandomDataset that tracks on_epoch_start calls via class-level registry."""

    _all_instances: list["EpochCountingDataset"] = []

    def __init__(self, config, data_location):
        super().__init__(config, data_location)
        self.epoch_start_count = 0
        self.last_verb: str | None = None
        EpochCountingDataset._all_instances.append(self)

    def on_epoch_start(self, verb):
        self.epoch_start_count += 1
        self.last_verb = verb

    def augment_image(self, data, idx, rng_seed):
        return -data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hyrax_config():
    h = hyrax.Hyrax()
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 10
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 42
    return h.config


def _make_dp(config, dataset_class, data_location, *, augment=None, fields=None):
    """Build a DataProvider with a single 'data' friendly name."""
    entry = {
        "dataset_class": dataset_class,
        "data_location": str(data_location),
        "primary_id_field": "object_id",
    }
    if augment is not None:
        entry["augment"] = augment
    if fields is not None:
        entry["fields"] = fields
    return DataProvider(config, {"data": entry})


# ---------------------------------------------------------------------------
# Step 2: HyraxDataset.on_epoch_start (base class no-op)
# ---------------------------------------------------------------------------


def test_hyrax_dataset_on_epoch_start_noop():
    """on_epoch_start is callable on the base class and is a no-op."""
    d = MinimalHyraxDataset(config={})
    d.on_epoch_start("train")  # must not raise


# ---------------------------------------------------------------------------
# Step 3: DataProvider augmentation dispatch
# ---------------------------------------------------------------------------


def test_augment_disabled_uses_only_getters(tmp_path):
    """With augment=False, only get_<field> methods are used."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=False, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    result = dp.resolve_data(0)
    assert dataset_instance.image_getter_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 0
    assert np.all(result["data"]["image"] >= 0)


def test_augment_absent_uses_only_getters(tmp_path):
    """With augment absent, only get_<field> methods are used."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    assert dataset_instance.image_getter_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 0


def test_augment_enabled_calls_augment_image(tmp_path):
    """With augment=True, augment_image is called and transforms the data."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=True, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    result = dp.resolve_data(0)
    assert len(dataset_instance.augment_image_calls) == 1
    assert np.all(result["data"]["image"] <= 0)


def test_augment_fallback_for_field_without_augment_method(tmp_path):
    """Fields without augment_<field> fall back to get_<field> when augment=True."""
    config = _make_hyrax_config()
    # AugmentedRandomDataset has augment_image but not augment_label
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=True, fields=["image", "label"])
    dataset_instance = dp.prepped_datasets["data"]

    result = dp.resolve_data(0)
    assert len(dataset_instance.augment_image_calls) == 1
    # image was negated by augment_image
    assert np.all(result["data"]["image"] <= 0)
    # label was not augmented — matches the raw get_label result
    assert result["data"]["label"] == dataset_instance.get_label(0)


def test_augment_rng_seed_same_within_row(tmp_path):
    """All augment_<field> calls for the same row receive the same rng_seed."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "SeedTrackingDataset", tmp_path, augment=True, fields=["image", "label"])
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    seeds = dataset_instance.seeds_by_idx[0]
    assert "image" in seeds and "label" in seeds
    assert seeds["image"] == seeds["label"]


def test_augment_rng_seed_differs_between_indices(tmp_path):
    """Different indices produce different rng_seeds."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "SeedTrackingDataset", tmp_path, augment=True, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    dp.resolve_data(1)
    assert dataset_instance.seeds_by_idx[0]["image"] != dataset_instance.seeds_by_idx[1]["image"]


def test_on_epoch_start_dispatches_to_all_datasets(tmp_path):
    """DataProvider.on_epoch_start calls on_epoch_start on every dataset instance."""
    config = _make_hyrax_config()
    dp = DataProvider(
        config,
        {
            "data": {
                "dataset_class": "AugmentedRandomDataset",
                "data_location": str(tmp_path),
                "primary_id_field": "object_id",
                "augment": True,
                "fields": ["image"],
            },
            "data2": {
                "dataset_class": "AugmentedRandomDataset",
                "data_location": str(tmp_path),
                "augment": True,
                "fields": ["image"],
            },
        },
    )
    instance1 = dp.prepped_datasets["data"]
    instance2 = dp.prepped_datasets["data2"]

    assert instance1.epoch_start_count == 0
    assert instance2.epoch_start_count == 0
    dp.on_epoch_start("train")
    assert instance1.epoch_start_count == 1
    assert instance2.epoch_start_count == 1
    dp.on_epoch_start("train")
    assert instance1.epoch_start_count == 2
    assert instance2.epoch_start_count == 2


def test_on_epoch_start_increments_epoch_counter(tmp_path):
    """DataProvider.on_epoch_start increments _current_epoch."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=True)
    assert dp._current_epoch == 0
    dp.on_epoch_start("train")
    assert dp._current_epoch == 1
    dp.on_epoch_start("train")
    assert dp._current_epoch == 2


def test_augment_cache_get_field_cached_augment_reruns(tmp_path):
    """With use_cache=True and augment=True: get_image runs once but augment_image runs every call."""
    config = _make_hyrax_config()
    # use_cache defaults to True
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=True, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    assert dataset_instance.image_getter_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 1

    # Second call: cache hit — get_image NOT called again, augment_image IS called again
    dp.resolve_data(0)
    assert dataset_instance.image_getter_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 2


def test_augment_list_dispatches_per_field(tmp_path):
    """augment list selectively enables augmentation per field."""
    config = _make_hyrax_config()
    dp = _make_dp(
        config,
        "AugmentedRandomDataset",
        tmp_path,
        augment=["image"],
        fields=["image", "label"],
    )
    dataset_instance = dp.prepped_datasets["data"]

    result = dp.resolve_data(0)
    # image was augmented (negated)
    assert len(dataset_instance.augment_image_calls) == 1
    assert np.all(result["data"]["image"] <= 0)
    # label was NOT augmented
    assert result["data"]["label"] == dataset_instance.get_label(0)


def test_augment_empty_list_no_augmentation(tmp_path):
    """augment as an empty list performs no augmentation."""
    config = _make_hyrax_config()
    dp = _make_dp(
        config,
        "AugmentedRandomDataset",
        tmp_path,
        augment=[],
        fields=["image"],
    )
    dataset_instance = dp.prepped_datasets["data"]

    result = dp.resolve_data(0)
    assert len(dataset_instance.augment_image_calls) == 0
    assert np.all(result["data"]["image"] >= 0)


def test_augment_list_missing_method_raises_runtime_error(tmp_path):
    """augment list naming a field without augment_<field> raises RuntimeError."""
    config = _make_hyrax_config()
    # AugmentedRandomDataset has augment_image but NOT augment_label
    with pytest.raises(RuntimeError, match="augment_label"):
        _make_dp(
            config,
            "AugmentedRandomDataset",
            tmp_path,
            augment=["image", "label"],
            fields=["image", "label"],
        )


def test_augment_list_rng_seed_same_within_row(tmp_path):
    """List-mode augmentation still passes the same rng_seed to all augment calls in a row."""
    config = _make_hyrax_config()
    dp = _make_dp(
        config,
        "SeedTrackingDataset",
        tmp_path,
        augment=["image", "label"],
        fields=["image", "label"],
    )
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    seeds = dataset_instance.seeds_by_idx[0]
    assert seeds["image"] == seeds["label"]


def test_no_augment_caching_unchanged(tmp_path):
    """Without augmentation, caching is identical to baseline (get_image called once per idx)."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=False, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    dp.resolve_data(0)
    assert dataset_instance.image_getter_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 0


# ---------------------------------------------------------------------------
# Augment list with fields not specified (implicit all-fields)
# ---------------------------------------------------------------------------


def test_augment_list_without_fields_happy_path(tmp_path):
    """augment list works when fields is not specified (all fields auto-discovered)."""
    config = _make_hyrax_config()
    # No fields= argument — all get_* methods are discovered automatically.
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    result = dp.resolve_data(0)
    # image was augmented (negated)
    assert len(dataset_instance.augment_image_calls) == 1
    assert np.all(result["data"]["image"] <= 0)
    # label was NOT augmented — uses get_label
    assert result["data"]["label"] == dataset_instance.get_label(0)


def test_augment_list_without_fields_no_augment_method(tmp_path):
    """augment list naming a field with no augment method raises RuntimeError when fields is omitted."""
    config = _make_hyrax_config()
    # AugmentedRandomDataset has get_label but NOT augment_label
    with pytest.raises(RuntimeError, match="augment_label"):
        _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=["label"])


def test_augment_list_without_fields_field_not_in_dataset(tmp_path):
    """augment list naming a nonexistent field raises RuntimeError when fields is omitted."""
    config = _make_hyrax_config()
    # "nonexistent" has no get_ or augment_ method on the dataset
    with pytest.raises(RuntimeError, match="not a field"):
        _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=["nonexistent"])


# ---------------------------------------------------------------------------
# Step 5: Integration — on_epoch_start wired to training loop
# ---------------------------------------------------------------------------


def test_on_epoch_start_called_during_training(tmp_path_factory):
    """on_epoch_start is dispatched once per training epoch for all DataProviders."""
    EpochCountingDataset._all_instances.clear()

    results_dir = tmp_path_factory.mktemp("epoch_count_results")
    data_dir = tmp_path_factory.mktemp("epoch_count_data")
    val_dir = tmp_path_factory.mktemp("epoch_count_val")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 2
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(results_dir)
    h.config["general"]["dev_mode"] = True
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 10
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0

    weights_file = results_dir / "fakeweights"
    weights_file.touch()
    h.config["infer"]["model_weights_file"] = str(weights_file)

    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "EpochCountingDataset",
                "data_location": str(data_dir),
                "primary_id_field": "object_id",
                "augment": True,
                "split_fraction": 0.8,
            }
        },
        "validate": {
            "data": {
                "dataset_class": "EpochCountingDataset",
                "data_location": str(val_dir),
                "primary_id_field": "object_id",
                "split_fraction": 0.2,
            }
        },
    }

    h.train()

    assert len(EpochCountingDataset._all_instances) >= 1
    # At least one DataProvider's dataset should have had on_epoch_start called 2 times
    assert any(inst.epoch_start_count == 2 for inst in EpochCountingDataset._all_instances)
    assert all(inst.last_verb == "train" for inst in EpochCountingDataset._all_instances)


# ---------------------------------------------------------------------------
# Step 6: verb argument propagation and dispatch from infer/test verbs
# ---------------------------------------------------------------------------


def test_on_epoch_start_passes_verb_to_dataset(tmp_path):
    """verb argument propagates from DataProvider.on_epoch_start down to each dataset instance."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=True)
    dataset_instance = dp.prepped_datasets["data"]

    dp.on_epoch_start("train")
    assert dataset_instance.last_verb == "train"

    dp.on_epoch_start("infer")
    assert dataset_instance.last_verb == "infer"


def test_on_epoch_start_called_during_infer(tmp_path_factory):
    """on_epoch_start is dispatched once during infer with verb='infer'."""
    EpochCountingDataset._all_instances.clear()

    results_dir = tmp_path_factory.mktemp("infer_epoch_results")
    data_dir = tmp_path_factory.mktemp("infer_epoch_data")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(results_dir)
    h.config["general"]["dev_mode"] = True
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 10
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0

    weights_file = results_dir / "fakeweights"
    weights_file.touch()
    h.config["infer"]["model_weights_file"] = str(weights_file)

    h.config["data_request"] = {
        "infer": {
            "data": {
                "dataset_class": "EpochCountingDataset",
                "data_location": str(data_dir),
                "primary_id_field": "object_id",
            }
        }
    }

    h.infer()

    assert len(EpochCountingDataset._all_instances) >= 1
    assert any(inst.epoch_start_count == 1 for inst in EpochCountingDataset._all_instances)
    assert all(inst.last_verb == "infer" for inst in EpochCountingDataset._all_instances)


def test_on_epoch_start_called_during_test(tmp_path_factory):
    """on_epoch_start is dispatched once during test with verb='test'."""
    from hyrax.config_utils import find_most_recent_results_dir

    EpochCountingDataset._all_instances.clear()

    results_dir = tmp_path_factory.mktemp("test_epoch_results")
    train_dir = tmp_path_factory.mktemp("test_epoch_train")
    val_dir = tmp_path_factory.mktemp("test_epoch_val")
    test_dir = tmp_path_factory.mktemp("test_epoch_test")
    infer_dir = tmp_path_factory.mktemp("test_epoch_infer")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(results_dir)
    h.config["general"]["dev_mode"] = True
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 10
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0

    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(train_dir),
                "primary_id_field": "object_id",
                "split_fraction": 0.8,
            }
        },
        "validate": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(val_dir),
                "primary_id_field": "object_id",
                "split_fraction": 0.2,
            }
        },
        "test": {
            "data": {
                "dataset_class": "EpochCountingDataset",
                "data_location": str(test_dir),
                "primary_id_field": "object_id",
            }
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(infer_dir),
                "primary_id_field": "object_id",
            }
        },
    }

    h.train()

    trained_weights = find_most_recent_results_dir(h.config, "train") / "example_model.pth"
    h.config["test"]["model_weights_file"] = str(trained_weights)

    # Reset so only instances created by h.test() are visible below.
    EpochCountingDataset._all_instances.clear()
    h.test()

    assert len(EpochCountingDataset._all_instances) >= 1
    assert any(inst.epoch_start_count == 1 for inst in EpochCountingDataset._all_instances)
    assert all(inst.last_verb == "test" for inst in EpochCountingDataset._all_instances)


# ---------------------------------------------------------------------------
# V3 Cache Restructuring: row_cache_key and per-dataset DataCache
# ---------------------------------------------------------------------------


def test_row_cache_key_default_returns_idx():
    """Default row_cache_key(idx) returns np.int64(idx)."""
    d = MinimalHyraxDataset(config={})
    result = d.row_cache_key(5)
    assert result == np.int64(5)
    assert isinstance(result, np.int64)


def test_row_cache_key_default_with_rng_seed_returns_none():
    """Default row_cache_key(idx, rng_seed=...) returns None."""
    d = MinimalHyraxDataset(config={})
    assert d.row_cache_key(5, rng_seed=np.int64(42)) is None


def test_row_cache_key_subclass_override():
    """Subclass override of row_cache_key is respected."""
    config = _make_hyrax_config()
    d = CachingAugmentDataset(config, data_location="/tmp")
    assert d.row_cache_key(3) == np.int64(3)
    assert d.row_cache_key(3, rng_seed=np.int64(42)) == np.int64(3 * 1_000_000 + 42)


def test_datacache_per_dataset_try_fetch_and_insert(tmp_path):
    """DataCache per-dataset insert and try_fetch round-trip correctly."""
    from hyrax.datasets.data_cache import DataCache

    config = _make_hyrax_config()
    ds = HyraxRandomDataset(config, data_location=str(tmp_path))
    datasets = {"data": ds}
    cache = DataCache(config, datasets, augment_active={"data": False})

    data = {"image": np.array([1, 2, 3])}
    cache.insert("data", real_idx=0, rng_seed=None, data=data)

    fetched, already_aug = cache.try_fetch("data", real_idx=0)
    assert fetched is data
    assert already_aug is False


def test_datacache_augmented_two_level_lookup(tmp_path):
    """DataCache two-level lookup: augmented key first, then base key."""
    from hyrax.datasets.data_cache import DataCache

    config = _make_hyrax_config()
    ds = CachingAugmentDataset(config, data_location=str(tmp_path))
    datasets = {"data": ds}
    cache = DataCache(config, datasets, augment_active={"data": True})

    base_data = {"image": np.array([1, 2, 3])}
    aug_data = {"image": np.array([-1, -2, -3])}

    cache.insert("data", real_idx=0, rng_seed=None, data=base_data)
    cache.insert("data", real_idx=0, rng_seed=np.int64(42), data=aug_data)

    # Augmented key hit
    fetched, already_aug = cache.try_fetch("data", real_idx=0, rng_seed=np.int64(42))
    assert fetched is aug_data
    assert already_aug is True

    # Different rng_seed misses augmented but hits base
    fetched, already_aug = cache.try_fetch("data", real_idx=0, rng_seed=np.int64(99))
    assert fetched is base_data
    assert already_aug is False


def test_datacache_row_cache_key_none_skips_cache(tmp_path):
    """row_cache_key returning None causes data to never be cached."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "NoCacheDataset", tmp_path, augment=False, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    assert dataset_instance.get_image_call_count == 1

    # get_image called again because nothing was cached
    dp.resolve_data(0)
    assert dataset_instance.get_image_call_count == 2


def test_augmented_data_not_cached_by_default(tmp_path):
    """With default row_cache_key, augmented data is not cached but base data is."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=True, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    assert dataset_instance.get_image_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 1

    # Second call: base data cached (get_image not called), augment re-runs
    dp.resolve_data(0)
    assert dataset_instance.get_image_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 2


def test_mixed_datasets_augmented_and_not(tmp_path):
    """One dataset with augmentation, one without — both cache correctly."""
    config = _make_hyrax_config()
    dp = DataProvider(
        config,
        {
            "data": {
                "dataset_class": "AugmentedRandomDataset",
                "data_location": str(tmp_path),
                "primary_id_field": "object_id",
                "augment": True,
                "fields": ["image"],
            },
            "data2": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path),
                "fields": ["image"],
            },
        },
    )

    result1 = dp.resolve_data(0)
    assert result1["data"] is not None
    assert result1["data2"] is not None

    aug_instance = dp.prepped_datasets["data"]
    assert len(aug_instance.augment_image_calls) == 1

    # Second call: augment re-runs for "data", but "data2" is fully cached
    result2 = dp.resolve_data(0)
    assert len(aug_instance.augment_image_calls) == 2
    # Non-augmented dataset returns same cached data
    assert np.array_equal(result1["data2"]["image"], result2["data2"]["image"])
