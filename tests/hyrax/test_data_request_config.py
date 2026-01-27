import pytest
from pydantic import ValidationError

from hyrax.config_schemas import (
    DataRequestConfig,
    DataRequestDefinition,
    HyraxCifarDatasetConfig,
    HyraxRandomDatasetConfig,
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
        train={"dataset_class": "TrainDS", "primary_id_field": "id1"},
        validate={"dataset_class": "ValidateDS", "primary_id_field": "id2"},
        infer={"dataset_class": "InferDS", "primary_id_field": "id3"},
        custom_split={"dataset_class": "ExtraDS", "primary_id_field": "id4"},
    )

    assert isinstance(definition.train, DataRequestConfig)
    assert isinstance(definition.validate, DataRequestConfig)
    assert isinstance(definition.infer, DataRequestConfig)
    assert "custom_split" in definition.other_datasets
    assert definition.other_datasets["custom_split"].dataset_class == "ExtraDS"


def test_data_request_definition_as_dict_shape():
    """as_dict returns nested data blocks."""
    definition = DataRequestDefinition(
        train={"dataset_class": "TrainDS", "fields": ["a"], "primary_id_field": "id1"},
        custom={"dataset_class": "ExtraDS", "primary_id_field": "id2"},
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
            "primary_id_field": "object_id",
            "dataset_config": {"size": 10, "shape": [1, 2, 3], "seed": 123},
        },
        infer={"dataset_class": "HyraxRandomDataset", "fields": ["image"], "primary_id_field": "id"},
    )

    cm.set_config("data_request", definition)

    rendered = cm.config["data_request"]
    assert rendered["train"]["data"]["dataset_class"] == "HyraxRandomDataset"
    assert rendered["train"]["data"]["fields"] == ["image"]
    assert rendered["train"]["data"]["dataset_config"]["shape"] == [1, 2, 3]
    assert rendered["infer"]["data"]["dataset_class"] == "HyraxRandomDataset"


def test_config_manager_set_config_with_valid_dict():
    """ConfigManager.set_config validates and coerces valid dict data_request."""

    cm = ConfigManager()
    valid_dict = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "fields": ["image"],
            "primary_id_field": "object_id",
        }
    }

    cm.set_config("data_request", valid_dict)

    rendered = cm.config["data_request"]
    assert rendered["train"]["data"]["dataset_class"] == "HyraxRandomDataset"
    assert rendered["train"]["data"]["fields"] == ["image"]
    assert rendered["train"]["data"]["primary_id_field"] == "object_id"


def test_config_manager_set_config_with_invalid_data_accepts_as_is():
    """ConfigManager.set_config accepts invalid data as-is when validation fails."""

    cm = ConfigManager()
    # Invalid: missing required primary_id_field
    invalid_dict = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "fields": ["image"],
            # Missing primary_id_field - would fail DataRequestDefinition validation
        }
    }

    # Should not raise - validation error is suppressed
    cm.set_config("data_request", invalid_dict)

    # Invalid data is stored as-is without validation/coercion
    rendered = cm.config["data_request"]
    assert rendered == invalid_dict
    assert "train" in rendered
    assert rendered["train"]["dataset_class"] == "HyraxRandomDataset"


def test_config_manager_set_config_with_completely_invalid_structure():
    """ConfigManager.set_config accepts completely invalid structures when validation fails."""

    cm = ConfigManager()
    # Completely invalid structure that can't be validated
    invalid_data = {"random_key": "random_value", "nested": {"deeply": {"invalid": 123}}}

    # Should not raise - validation error is suppressed
    cm.set_config("data_request", invalid_data)

    # Invalid data is stored as-is
    rendered = cm.config["data_request"]
    assert rendered == invalid_data


def test_config_manager_set_config_coerces_typed_dataset_config():
    """ConfigManager.set_config properly coerces dataset_config for known datasets."""

    cm = ConfigManager()
    valid_dict = {
        "train": {
            "dataset_class": "HyraxCifarDataset",
            "primary_id_field": "id",
            "dataset_config": {"use_training_data": True},
        }
    }

    cm.set_config("data_request", valid_dict)

    rendered = cm.config["data_request"]
    assert rendered["train"]["data"]["dataset_class"] == "HyraxCifarDataset"
    assert rendered["train"]["data"]["dataset_config"]["use_training_data"] is True


def test_config_manager_set_config_with_partial_validity():
    """ConfigManager.set_config handles edge case with partially valid data."""

    cm = ConfigManager()
    # Has some valid structure but missing required fields in one split
    partially_valid = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "primary_id_field": "id",
        },
        "validate": {
            "dataset_class": "HyraxCifarDataset",
            # Missing primary_id_field - makes the whole definition invalid
        },
    }

    # Should not raise - validation error is suppressed
    cm.set_config("data_request", partially_valid)

    # Since validation fails, data is stored as-is
    rendered = cm.config["data_request"]
    assert rendered == partially_valid


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


def test_single_config_with_primary_id_valid():
    """Single config with primary_id_field passes validation."""
    definition = DataRequestDefinition(
        train=DataRequestConfig(
            dataset_class="HyraxRandomDataset",
            primary_id_field="object_id",
        )
    )
    assert definition.train.primary_id_field == "object_id"


def test_single_config_without_primary_id_fails():
    """Single config without primary_id_field fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            train=DataRequestConfig(
                dataset_class="HyraxRandomDataset",
            )
        )
    assert (
        "'train' must have exactly one DataRequestConfig with 'primary_id_field' set, but found none"
        in str(exc_info.value)
    )


def test_dict_configs_one_primary_id_valid():
    """Dict with exactly one primary_id_field passes validation."""
    definition = DataRequestDefinition(
        train={
            "data_0": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                primary_id_field="some_field",
            ),
            "data_1": DataRequestConfig(
                dataset_class="HyraxCifarDataset",
            ),
        }
    )
    assert isinstance(definition.train, dict)
    assert definition.train["data_0"].primary_id_field == "some_field"
    assert definition.train["data_1"].primary_id_field is None


def test_dict_configs_no_primary_id_fails():
    """Dict with no primary_id_field fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            train={
                "data_0": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                ),
                "data_1": DataRequestConfig(
                    dataset_class="HyraxCifarDataset",
                ),
            }
        )
    assert (
        "'train' must have exactly one DataRequestConfig with 'primary_id_field' set, but found none"
        in str(exc_info.value)
    )


def test_dict_configs_multiple_primary_ids_fails():
    """Dict with multiple primary_id_fields fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            train={
                "data_0": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    primary_id_field="field_0",
                ),
                "data_1": DataRequestConfig(
                    dataset_class="HyraxCifarDataset",
                    primary_id_field="field_1",
                ),
            }
        )
    assert "'train' must have exactly one DataRequestConfig with 'primary_id_field' set, but found 2" in str(
        exc_info.value
    )


def test_multiple_dataset_groups_each_validated():
    """Each dataset group is validated independently."""
    # Valid: each group has exactly one primary_id_field
    definition = DataRequestDefinition(
        train=DataRequestConfig(
            dataset_class="HyraxRandomDataset",
            primary_id_field="train_id",
        ),
        validate=DataRequestConfig(
            dataset_class="HyraxCifarDataset",
            primary_id_field="validate_id",
        ),
        infer=DataRequestConfig(
            dataset_class="HyraxRandomDataset",
            primary_id_field="infer_id",
        ),
    )
    assert definition.train.primary_id_field == "train_id"
    assert definition.validate.primary_id_field == "validate_id"
    assert definition.infer.primary_id_field == "infer_id"

    # Invalid: train missing primary_id_field
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            train=DataRequestConfig(
                dataset_class="HyraxRandomDataset",
            ),
            validate=DataRequestConfig(
                dataset_class="HyraxCifarDataset",
                primary_id_field="validate_id",
            ),
        )
    assert (
        "'train' must have exactly one DataRequestConfig with 'primary_id_field' set, but found none"
        in str(exc_info.value)
    )


def test_dict_key_names_dont_matter():
    """Arbitrary friendly names are allowed for dict keys."""
    definition = DataRequestDefinition(
        train={
            "my_custom_name": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                primary_id_field="id",
            ),
            "another_name": DataRequestConfig(
                dataset_class="HyraxCifarDataset",
            ),
        }
    )
    assert "my_custom_name" in definition.train
    assert "another_name" in definition.train


def test_from_dict_format():
    """Construction from dictionary format works correctly."""
    config_dict = {
        "train": {
            "dataset_0": {
                "dataset_class": "HyraxRandomDataset",
                "primary_id_field": "obj_id",
            },
            "dataset_1": {
                "dataset_class": "HyraxCifarDataset",
            },
        }
    }
    definition = DataRequestDefinition.model_validate(config_dict)
    assert isinstance(definition.train, dict)
    assert definition.train["dataset_0"].primary_id_field == "obj_id"
    assert definition.train["dataset_1"].primary_id_field is None


def test_as_dict_with_single_config():
    """as_dict handles single config correctly."""
    definition = DataRequestDefinition(
        train=DataRequestConfig(
            dataset_class="HyraxRandomDataset",
            primary_id_field="id",
        )
    )
    as_dict = definition.as_dict()
    assert "train" in as_dict
    assert "data" in as_dict["train"]
    assert as_dict["train"]["data"]["dataset_class"] == "HyraxRandomDataset"
    assert as_dict["train"]["data"]["primary_id_field"] == "id"


def test_as_dict_with_dict_configs():
    """as_dict handles dict of configs correctly."""
    definition = DataRequestDefinition(
        train={
            "data_0": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                primary_id_field="id",
            ),
            "data_1": DataRequestConfig(
                dataset_class="HyraxCifarDataset",
            ),
        }
    )
    as_dict = definition.as_dict()
    assert "train" in as_dict
    assert "data_0" in as_dict["train"]
    assert "data_1" in as_dict["train"]
    assert as_dict["train"]["data_0"]["data"]["dataset_class"] == "HyraxRandomDataset"
    assert as_dict["train"]["data_0"]["data"]["primary_id_field"] == "id"
    assert as_dict["train"]["data_1"]["data"]["dataset_class"] == "HyraxCifarDataset"


def test_other_datasets_validation():
    """other_datasets field is also validated for primary_id_field."""
    # Valid
    definition = DataRequestDefinition(
        custom_split=DataRequestConfig(
            dataset_class="HyraxRandomDataset",
            primary_id_field="id",
        )
    )
    assert "custom_split" in definition.other_datasets

    # Invalid - no primary_id_field
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            custom_split=DataRequestConfig(
                dataset_class="HyraxRandomDataset",
            )
        )
    assert (
        "Dataset 'custom_split' must have exactly one DataRequestConfig "
        "with 'primary_id_field' set, but found none" in str(exc_info.value)
    )
