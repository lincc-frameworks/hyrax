# ruff: noqa: D102
"""Tests for the augmentation feature."""

import numpy as np

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
        self.get_image_call_count = 0
        self.augment_image_calls: list[tuple[int, int]] = []
        self.epoch_start_count = 0

    def on_epoch_start(self):
        self.epoch_start_count += 1

    def get_image(self, idx):
        self.get_image_call_count += 1
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


class EpochCountingDataset(HyraxRandomDataset):
    """HyraxRandomDataset that tracks on_epoch_start calls via class-level registry."""

    _all_instances: list["EpochCountingDataset"] = []

    def __init__(self, config, data_location):
        super().__init__(config, data_location)
        self.epoch_start_count = 0
        EpochCountingDataset._all_instances.append(self)

    def on_epoch_start(self):
        self.epoch_start_count += 1

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
    d.on_epoch_start()  # must not raise


# ---------------------------------------------------------------------------
# Step 3: DataProvider augmentation dispatch
# ---------------------------------------------------------------------------


def test_augment_disabled_uses_only_getters(tmp_path):
    """With augment=False, only get_<field> methods are used."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=False, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    result = dp.resolve_data(0)
    assert dataset_instance.get_image_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 0
    assert np.all(result["data"]["image"] >= 0)


def test_augment_absent_uses_only_getters(tmp_path):
    """With augment absent, only get_<field> methods are used."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    assert dataset_instance.get_image_call_count == 1
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
    dp.on_epoch_start()
    assert instance1.epoch_start_count == 1
    assert instance2.epoch_start_count == 1
    dp.on_epoch_start()
    assert instance1.epoch_start_count == 2
    assert instance2.epoch_start_count == 2


def test_on_epoch_start_increments_epoch_counter(tmp_path):
    """DataProvider.on_epoch_start increments _current_epoch."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=True)
    assert dp._current_epoch == 0
    dp.on_epoch_start()
    assert dp._current_epoch == 1
    dp.on_epoch_start()
    assert dp._current_epoch == 2


def test_augment_cache_get_field_cached_augment_reruns(tmp_path):
    """With use_cache=True and augment=True: get_image runs once but augment_image runs every call."""
    config = _make_hyrax_config()
    # use_cache defaults to True
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=True, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    assert dataset_instance.get_image_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 1

    # Second call: cache hit — get_image NOT called again, augment_image IS called again
    dp.resolve_data(0)
    assert dataset_instance.get_image_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 2


def test_no_augment_caching_unchanged(tmp_path):
    """Without augmentation, caching is identical to baseline (get_image called once per idx)."""
    config = _make_hyrax_config()
    dp = _make_dp(config, "AugmentedRandomDataset", tmp_path, augment=False, fields=["image"])
    dataset_instance = dp.prepped_datasets["data"]

    dp.resolve_data(0)
    dp.resolve_data(0)
    assert dataset_instance.get_image_call_count == 1
    assert len(dataset_instance.augment_image_calls) == 0


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
