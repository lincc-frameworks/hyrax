from pathlib import Path

import pytest
from pydantic import ValidationError

from hyrax.config_schemas import (
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
    assert dumped["data_location"] == str(Path("/tmp/data").expanduser().resolve())
    assert dumped["fields"] == ["image", "label"]
    assert dumped["primary_id_field"] == "object_id"
    assert dumped["dataset_config"] == {"shuffle": True}


def test_data_request_config_unwraps_data_key():
    """Support legacy wrapped 'data' key."""
    cfg = DataRequestConfig(
        data={"dataset_class": "HyraxCifarDataset", "primary_id_field": "oid", "data_location": "nowhere"}
    )
    assert cfg.dataset_class == "HyraxCifarDataset"
    assert cfg.primary_id_field == "oid"


def test_data_request_definition_collects_known_fields():
    """Collect train/validate/infer fields."""
    definition = DataRequestDefinition(
        {
            "train": {"dataset_class": "TrainDS", "primary_id_field": "id1", "data_location": "nowhere"},
            "validate": {
                "dataset_class": "ValidateDS",
                "primary_id_field": "id2",
                "data_location": "nowhere",
            },
            "infer": {"dataset_class": "InferDS", "primary_id_field": "id3", "data_location": "nowhere"},
        }
    )

    assert isinstance(definition["train"], DataRequestConfig)
    assert isinstance(definition["validate"], DataRequestConfig)
    assert isinstance(definition["infer"], DataRequestConfig)


def test_data_request_definition_as_dict_shape():
    """as_dict returns nested data blocks."""
    definition = DataRequestDefinition(
        {
            "train": {
                "dataset_class": "TrainDS",
                "fields": ["a"],
                "primary_id_field": "id1",
                "data_location": "nowhere",
            },
            "validate": {
                "dataset_class": "ExtraDS",
                "primary_id_field": "id2",
                "data_location": "nowhere",
            },
        }
    )

    as_dict = definition.as_dict()
    assert set(as_dict.keys()) == {"train", "validate"}
    assert as_dict["train"]["data"]["dataset_class"] == "TrainDS"
    assert as_dict["train"]["data"]["fields"] == ["a"]
    assert as_dict["validate"]["data"]["dataset_class"] == "ExtraDS"


def test_config_manager_set_config_accepts_data_request_definition():
    """ConfigManager.set_config should accept pydantic model for data_request."""

    cm = ConfigManager()
    definition = DataRequestDefinition(
        {
            "train": {
                "dataset_class": "HyraxRandomDataset",
                "fields": ["image"],
                "primary_id_field": "object_id",
                "data_location": "nowhere",
                "dataset_config": {"size": 10, "shape": [1, 2, 3], "seed": 123},
            },
            "infer": {
                "dataset_class": "HyraxRandomDataset",
                "fields": ["image"],
                "primary_id_field": "id",
                "data_location": "somewhere",
            },
        }
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
            "data_location": "nowhere",
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
    # Invalid: missing primary_id_field required by DataRequestDefinition validation
    invalid_dict = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "fields": ["image"],
            "data_location": "nowhere",
            # Missing primary_id_field - fails DataRequestDefinition validation
        }
    }

    # Should not raise - validation error is logged as warning but data is accepted
    cm.set_config("data_request", invalid_dict)

    # Invalid data is stored as-is without validation/coercion
    rendered = cm.config["data_request"]
    assert rendered == invalid_dict
    assert "train" in rendered
    assert rendered["train"]["dataset_class"] == "HyraxRandomDataset"


def test_config_manager_set_config_coerces_typed_dataset_config():
    """ConfigManager.set_config properly coerces dataset_config for known datasets."""

    cm = ConfigManager()
    valid_dict = {
        "train": {
            "dataset_class": "HyraxCifarDataset",
            "data_location": "nowhere",
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
    # Has some valid structure but missing primary_id_field in one split
    partially_valid = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "nowhere",
            "primary_id_field": "id",
        },
        "validate": {
            "dataset_class": "HyraxCifarDataset",
            "data_location": "nowhere",
            # Missing primary_id_field - fails DataRequestDefinition validation requirement
        },
    }

    # Should not raise - validation error is logged as warning but data is accepted
    cm.set_config("data_request", partially_valid)

    # Since validation fails, data is stored as-is
    rendered = cm.config["data_request"]
    assert rendered == partially_valid


def test_data_request_definition_rejects_missing_dataset_class():
    """dataset_class is required."""
    with pytest.raises(ValidationError):
        DataRequestConfig(data_location="/tmp/data")


def test_dataset_config_unknown_dataset_allows_dict():
    """Unknown dataset_class leaves dataset_config as plain dict."""
    cfg = DataRequestConfig(
        dataset_class="MyCustomDataset", data_location="/dev/null", dataset_config={"foo": "bar"}
    )

    assert cfg.dataset_config == {"foo": "bar"}


def test_single_config_with_primary_id_valid():
    """Single config with primary_id_field passes validation."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/dev/null",
                primary_id_field="object_id",
            )
        }
    )
    assert definition["train"].primary_id_field == "object_id"


def test_single_config_without_primary_id_fails():
    """Single config without primary_id_field fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            {
                "train": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="somewhere",
                )
            }
        )
    assert (
        "'train' must have exactly one DataRequestConfig with 'primary_id_field' set, but found none"
        in str(exc_info.value)
    )


def test_dict_configs_one_primary_id_valid():
    """Dict with exactly one primary_id_field passes validation."""
    definition = DataRequestDefinition(
        {
            "train": {
                "data_0": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="somewhere",
                    primary_id_field="some_field",
                ),
                "data_1": DataRequestConfig(
                    dataset_class="HyraxCifarDataset",
                    data_location="somewhere",
                ),
            }
        }
    )
    assert isinstance(definition["train"], dict)
    assert definition["train"]["data_0"].primary_id_field == "some_field"
    assert definition["train"]["data_1"].primary_id_field is None


def test_dict_configs_no_primary_id_fails():
    """Dict with no primary_id_field fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            {
                "train": {
                    "data_0": DataRequestConfig(
                        dataset_class="HyraxRandomDataset",
                        data_location="somewhere",
                    ),
                    "data_1": DataRequestConfig(
                        dataset_class="HyraxCifarDataset",
                        data_location="somewhere",
                    ),
                }
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
            {
                "train": {
                    "data_0": DataRequestConfig(
                        dataset_class="HyraxRandomDataset",
                        data_location="somewhere",
                        primary_id_field="field_0",
                    ),
                    "data_1": DataRequestConfig(
                        dataset_class="HyraxCifarDataset",
                        data_location="somewhere",
                        primary_id_field="field_1",
                    ),
                }
            }
        )
    assert "'train' must have exactly one DataRequestConfig with 'primary_id_field' set, but found 2" in str(
        exc_info.value
    )


def test_multiple_dataset_groups_each_validated():
    """Each dataset group is validated independently."""
    # Valid: each group has exactly one primary_id_field
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="somewhere",
                primary_id_field="train_id",
            ),
            "validate": DataRequestConfig(
                dataset_class="HyraxCifarDataset",
                data_location="somewhere",
                primary_id_field="validate_id",
            ),
            "infer": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="somewhere",
                primary_id_field="infer_id",
            ),
        }
    )
    assert definition["train"].primary_id_field == "train_id"
    assert definition["validate"].primary_id_field == "validate_id"
    assert definition["infer"].primary_id_field == "infer_id"

    # Invalid: train missing primary_id_field
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            {
                "train": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="somewhere",
                ),
                "validate": DataRequestConfig(
                    dataset_class="HyraxCifarDataset",
                    data_location="somewhere",
                    primary_id_field="validate_id",
                ),
            }
        )
    assert (
        "'train' must have exactly one DataRequestConfig with 'primary_id_field' set, but found none"
        in str(exc_info.value)
    )


def test_dict_key_names_dont_matter():
    """Arbitrary friendly names are allowed for dict keys."""
    definition = DataRequestDefinition(
        {
            "train": {
                "my_custom_name": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="somewhere",
                    primary_id_field="id",
                ),
                "another_name": DataRequestConfig(
                    dataset_class="HyraxCifarDataset",
                    data_location="somewhere",
                ),
            }
        }
    )
    assert "my_custom_name" in definition["train"]
    assert "another_name" in definition["train"]


def test_from_dict_format():
    """Construction from dictionary format works correctly."""
    config_dict = {
        "train": {
            "dataset_0": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "/dev/null",
                "primary_id_field": "obj_id",
            },
            "dataset_1": {
                "dataset_class": "HyraxCifarDataset",
                "data_location": "somewhere",
            },
        }
    }
    definition = DataRequestDefinition.model_validate(config_dict)
    assert isinstance(definition["train"], dict)
    assert definition["train"]["dataset_0"].primary_id_field == "obj_id"
    assert definition["train"]["dataset_1"].primary_id_field is None


def test_as_dict_with_single_config():
    """as_dict handles single config correctly."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="somewhere",
                primary_id_field="id",
            )
        }
    )
    as_dict = definition.as_dict()
    assert "train" in as_dict
    assert "data" in as_dict["train"]
    assert as_dict["train"]["data"]["dataset_class"] == "HyraxRandomDataset"
    assert as_dict["train"]["data"]["primary_id_field"] == "id"


def test_as_dict_with_dict_configs():
    """as_dict handles dict of configs correctly."""
    definition = DataRequestDefinition(
        {
            "train": {
                "data_0": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="somewhere",
                    primary_id_field="id",
                ),
                "data_1": DataRequestConfig(
                    dataset_class="HyraxCifarDataset",
                    data_location="somewhere",
                ),
            }
        }
    )
    as_dict = definition.as_dict()
    assert "train" in as_dict
    assert "data_0" in as_dict["train"]
    assert "data_1" in as_dict["train"]
    assert as_dict["train"]["data_0"]["data"]["dataset_class"] == "HyraxRandomDataset"
    assert as_dict["train"]["data_0"]["data"]["primary_id_field"] == "id"
    assert as_dict["train"]["data_1"]["data"]["dataset_class"] == "HyraxCifarDataset"


def test_split_fraction_not_present():
    """split_fraction defaults to None and raises no errors when omitted."""
    cfg = DataRequestConfig(
        dataset_class="HyraxRandomDataset",
        data_location="/dev/null",
        primary_id_field="id",
    )
    assert cfg.split_fraction is None


@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.99, 1.0])
def test_split_fraction_valid_values(fraction):
    """split_fraction accepts values in the range (0.0, 1.0]."""
    cfg = DataRequestConfig(
        dataset_class="HyraxRandomDataset",
        data_location="/dev/null",
        primary_id_field="id",
        split_fraction=fraction,
    )
    assert cfg.split_fraction == fraction


@pytest.mark.parametrize("fraction", [0.0, -0.1, -1.0, 1.01, 2.0, 100.0])
def test_split_fraction_invalid_values(fraction):
    """split_fraction rejects values outside (0.0, 1.0]."""
    with pytest.raises(ValidationError):
        DataRequestConfig(
            dataset_class="HyraxRandomDataset",
            data_location="/dev/null",
            primary_id_field="id",
            split_fraction=fraction,
        )


def test_data_location_resolves_relative_path():
    """Relative data_location paths are fully resolved to absolute paths."""
    cfg = DataRequestConfig(
        dataset_class="HyraxRandomDataset",
        data_location="./some/relative/path",
        primary_id_field="id",
    )
    resolved = cfg.data_location
    expected = str(Path("./some/relative/path").expanduser().resolve())
    assert resolved == expected
    assert Path(resolved).is_absolute()


def test_data_location_unchanged_when_absolute():
    """Absolute data_location paths remain unchanged after validation."""
    cfg = DataRequestConfig(
        dataset_class="HyraxRandomDataset",
        data_location="/tmp/absolute/path",
        primary_id_field="id",
    )
    assert cfg.data_location == str(Path("/tmp/absolute/path").expanduser().resolve())


def test_split_fraction_sum_valid_same_location():
    """Split fractions for the same data_location summing to <= 1.0 pass validation."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data",
                primary_id_field="id",
                split_fraction=0.6,
            ),
            "validate": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data",
                primary_id_field="id",
                split_fraction=0.4,
            ),
        }
    )
    assert definition["train"].split_fraction == 0.6
    assert definition["validate"].split_fraction == 0.4


def test_split_fraction_sum_fp_rounding_at_one_passes():
    """Floating-point rounding at 1.0 does not trigger a false validation error."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data",
                primary_id_field="id",
                split_fraction=0.1,
            ),
            "validate": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data",
                primary_id_field="id",
                split_fraction=0.2,
            ),
            "infer": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data",
                primary_id_field="id",
                split_fraction=0.7,
            ),
        }
    )
    assert definition["train"].split_fraction == 0.1
    assert definition["validate"].split_fraction == 0.2
    assert definition["infer"].split_fraction == 0.7


def test_split_fraction_sum_valid_different_locations():
    """Split fractions for different data_locations are validated independently."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data_a",
                primary_id_field="id",
                split_fraction=0.8,
            ),
            "validate": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data_b",
                primary_id_field="id",
                split_fraction=0.9,
            ),
        }
    )
    assert definition["train"].split_fraction == 0.8
    assert definition["validate"].split_fraction == 0.9


def test_split_fraction_sum_exceeds_one_same_location():
    """Split fractions for the same data_location summing to > 1.0 fail validation."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            {
                "train": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/tmp/data",
                    primary_id_field="id",
                    split_fraction=0.7,
                ),
                "validate": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/tmp/data",
                    primary_id_field="id",
                    split_fraction=0.5,
                ),
            }
        )
    assert "exceeds 1.0" in str(exc_info.value)


def test_split_fraction_sum_with_none_fractions_different_locations():
    """Configs without split_fraction at a different location don't affect the sum."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data_a",
                primary_id_field="id",
                split_fraction=0.8,
            ),
            "validate": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data_b",
                primary_id_field="id",
                # No split_fraction — different location, so no conflict
            ),
        }
    )
    assert definition["train"].split_fraction == 0.8
    assert definition["validate"].split_fraction is None


def test_split_fraction_sum_across_three_groups():
    """Split fractions across train, validate, and infer for the same location are summed."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            {
                "train": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/tmp/data",
                    primary_id_field="id",
                    split_fraction=0.4,
                ),
                "validate": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/tmp/data",
                    primary_id_field="id",
                    split_fraction=0.4,
                ),
                "infer": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/tmp/data",
                    primary_id_field="id",
                    split_fraction=0.3,
                ),
            }
        )
    assert "exceeds 1.0" in str(exc_info.value)


def test_split_fraction_sum_dict_configs_same_location():
    """Split fractions across groups sharing the same location are summed."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            {
                "train": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/tmp/data",
                    primary_id_field="id",
                    split_fraction=0.6,
                ),
                "infer": DataRequestConfig(
                    dataset_class="HyraxCifarDataset",
                    data_location="/tmp/data",
                    primary_id_field="id",
                    split_fraction=0.5,
                ),
            }
        )
    assert "exceeds 1.0" in str(exc_info.value)


def test_split_fraction_requires_primary_id_field():
    """split_fraction cannot be set without primary_id_field."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestConfig(
            dataset_class="HyraxRandomDataset",
            data_location="/dev/null",
            split_fraction=0.5,
        )
    assert "split_fraction" in str(exc_info.value)
    assert "primary_id_field" in str(exc_info.value)


def test_split_fraction_allowed_with_primary_id_field():
    """split_fraction is accepted when primary_id_field is also provided."""
    cfg = DataRequestConfig(
        dataset_class="HyraxRandomDataset",
        data_location="/dev/null",
        primary_id_field="id",
        split_fraction=0.5,
    )
    assert cfg.split_fraction == 0.5
    assert cfg.primary_id_field == "id"


def test_no_split_fraction_without_primary_id_is_fine():
    """Omitting both split_fraction and primary_id_field raises no error."""
    cfg = DataRequestConfig(
        dataset_class="HyraxRandomDataset",
        data_location="/dev/null",
    )
    assert cfg.split_fraction is None
    assert cfg.primary_id_field is None


def test_arbitrary_group_names_accepted():
    """DataRequestDefinition accepts arbitrary group names like 'test' or 'finetune'."""
    definition = DataRequestDefinition(
        {
            "test": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/dev/null",
                primary_id_field="id",
            ),
            "finetune": DataRequestConfig(
                dataset_class="HyraxCifarDataset",
                data_location="/dev/null",
                primary_id_field="id",
            ),
        }
    )
    assert definition["test"].dataset_class == "HyraxRandomDataset"
    assert definition["finetune"].dataset_class == "HyraxCifarDataset"
    assert "test" in definition
    assert "finetune" in definition


def test_arbitrary_group_name_in_as_dict():
    """as_dict includes arbitrary group names in the output."""
    definition = DataRequestDefinition(
        {
            "test": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/dev/null",
                primary_id_field="id",
            ),
        }
    )
    as_dict = definition.as_dict()
    assert "test" in as_dict
    assert as_dict["test"]["data"]["dataset_class"] == "HyraxRandomDataset"


def test_missing_group_returns_none():
    """Accessing a non-existent group name returns None."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/dev/null",
                primary_id_field="id",
            ),
        }
    )
    assert definition.root.get("nonexistent") is None
    assert "nonexistent" not in definition


def test_empty_definition_raises():
    """An empty DataRequestDefinition raises a validation error."""
    with pytest.raises(ValidationError):
        DataRequestDefinition({})


def test_none_groups_are_skipped():
    """Groups set to None are excluded from the definition."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/dev/null",
                primary_id_field="id",
            ),
            "validate": None,
        }
    )
    assert definition["train"] is not None
    assert definition.root.get("validate") is None
    assert "validate" not in definition


def test_split_fraction_consistency_mixed_raises():
    """If one config has split_fraction for a location, all must."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            {
                "train": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/tmp/data",
                    primary_id_field="id",
                    split_fraction=0.5,
                ),
                "infer": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/tmp/data",
                    primary_id_field="id",
                    # Missing split_fraction
                ),
            }
        )
    assert "split_fraction" in str(exc_info.value)
    assert "infer" in str(exc_info.value)


def test_split_fraction_consistency_all_set_passes():
    """All configs sharing a location with split_fraction set passes."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data",
                primary_id_field="id",
                split_fraction=0.6,
            ),
            "infer": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data",
                primary_id_field="id",
                split_fraction=0.4,
            ),
        }
    )
    assert definition["train"].split_fraction == 0.6
    assert definition["infer"].split_fraction == 0.4


def test_split_fraction_consistency_none_set_passes():
    """No configs having split_fraction for a shared location passes."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data",
                primary_id_field="id",
            ),
            "infer": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data",
                primary_id_field="id",
            ),
        }
    )
    assert definition["train"].split_fraction is None
    assert definition["infer"].split_fraction is None


def test_split_fraction_consistency_single_config_no_issue():
    """A single config with split_fraction doesn't trigger the consistency check."""
    definition = DataRequestDefinition(
        {
            "train": DataRequestConfig(
                dataset_class="HyraxRandomDataset",
                data_location="/tmp/data",
                primary_id_field="id",
                split_fraction=0.7,
            ),
        }
    )
    assert definition["train"].split_fraction == 0.7
