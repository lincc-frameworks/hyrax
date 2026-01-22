import pytest
from pydantic import ValidationError

from hyrax.config_schemas import (
    HyraxCifarDatasetConfig,
    HyraxRandomDatasetConfig,
    DataRequestConfig,
    DataRequestDefinition,
)
from hyrax.config_utils import ConfigManager


def test_data_request_config_basic_fields():
    """Basic field round trip."""
    cfg = DataRequestConfig(
        dataset_class="hyrax.data_sets.hyrax_csv_dataset.HyraxCSVDataset",
        data_location="/tmp/data",
        fields=["image", "label"],
        primary_id_field="object_id",
        dataset_config={"shuffle": True},
    )

    dumped = cfg.as_dict()
    assert dumped["dataset_class"].endswith("HyraxCSVDataset")
    assert dumped["data_location"] == "/tmp/data"
    assert dumped["fields"] == ["image", "label"]
    assert dumped["primary_id_field"] == "object_id"
    assert dumped["dataset_config"] == {"shuffle": True}


def test_data_request_config_unwraps_data_key():
    """Support legacy wrapped 'data' key."""
    cfg = DataRequestConfig(data={"dataset_class": "HyraxCifarDataset", "primary_id_field": "oid"})
    assert cfg.dataset_class == "HyraxCifarDataset"
    assert cfg.primary_id_field == "oid"


def test_data_request_definition_collects_known_and_extra():
    """Collect train/validate/infer plus extra datasets."""
    definition = DataRequestDefinition(
        train={"dataset_class": "TrainDS"},
        validate={"dataset_class": "ValidateDS"},
        infer={"dataset_class": "InferDS"},
        custom_split={"dataset_class": "ExtraDS"},
    )

    assert isinstance(definition.train, DataRequestConfig)
    assert isinstance(definition.validate, DataRequestConfig)
    assert isinstance(definition.infer, DataRequestConfig)
    assert "custom_split" in definition.other_datasets
    assert definition.other_datasets["custom_split"].dataset_class == "ExtraDS"


def test_data_request_definition_as_dict_shape():
    """as_dict returns nested data blocks."""
    definition = DataRequestDefinition(
        train={"dataset_class": "TrainDS", "fields": ["a"]},
        custom={"dataset_class": "ExtraDS"},
    )

    as_dict = definition.as_dict()
    assert set(as_dict.keys()) == {"train", "custom"}
    assert as_dict["train"]["data"]["dataset_class"] == "TrainDS"
    assert as_dict["train"]["data"]["fields"] == ["a"]
    assert as_dict["custom"]["data"]["dataset_class"] == "ExtraDS"


def test_config_manager_set_config_accepts_data_request_definition():
    """ConfigManager.set_config should accept pydantic model for data_request."""

    cm = ConfigManager()
    definition = DataRequestDefinition(
        train={
            "dataset_class": "HyraxRandomDataset",
            "fields": ["image"],
            "dataset_config": {"size": 10, "shape": [1, 2, 3], "seed": 123},
        },
        infer={"dataset_class": "HyraxRandomDataset", "fields": ["image"]},
    )

    cm.set_config("data_request", definition)

    rendered = cm.config["data_request"]
    assert rendered["train"]["data"]["dataset_class"] == "HyraxRandomDataset"
    assert rendered["train"]["data"]["fields"] == ["image"]
    assert rendered["train"]["data"]["dataset_config"]["shape"] == [1, 2, 3]
    assert rendered["infer"]["data"]["dataset_class"] == "HyraxRandomDataset"


def test_data_request_definition_rejects_missing_dataset_class():
    """dataset_class is required."""
    with pytest.raises(ValidationError):
        DataRequestConfig(data_location="/tmp/data")


def test_dataset_config_typed_mapping_random():
    """dataset_config coerces to typed config for random dataset."""
    cfg = DataRequestConfig(
        dataset_class="HyraxRandomDataset",
        dataset_config={"size": 10, "shape": [1, 2], "seed": 123},
    )

    assert isinstance(cfg.dataset_config, HyraxRandomDatasetConfig)
    assert cfg.dataset_config.size == 10
    assert cfg.dataset_config.shape == [1, 2]
    assert cfg.dataset_config.seed == 123


def test_dataset_config_typed_mapping_cifar():
    """dataset_config coerces to typed config for cifar dataset."""
    cfg = DataRequestConfig(
        dataset_class="HyraxCifarDataset",
        dataset_config={"use_training_data": False},
    )

    assert isinstance(cfg.dataset_config, HyraxCifarDatasetConfig)
    assert cfg.dataset_config.use_training_data is False


def test_dataset_config_unknown_dataset_allows_dict():
    """Unknown dataset_class leaves dataset_config as plain dict."""
    cfg = DataRequestConfig(dataset_class="MyCustomDataset", dataset_config={"foo": "bar"})

    assert cfg.dataset_config == {"foo": "bar"}
