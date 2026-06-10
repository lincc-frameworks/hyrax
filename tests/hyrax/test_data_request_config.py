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
        dataset_class="hyrax.datasets.hyrax_csv_dataset.HyraxCSVDataset",
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


def test_data_request_config_preserves_hf_uri_data_location():
    """Test that HF URIs are preserved when they're used as a data location"""
    cfg = DataRequestConfig(
        dataset_class="MultimodalUniverseDataset",
        data_location="hf://MultimodalUniverse/plasticc",
        primary_id_field="object_id",
    )
    assert cfg.data_location == "hf://MultimodalUniverse/plasticc"


def test_data_request_config_preserves_http_uri_data_location():
    """Test that HTTP URIs are preserved when they're used as a data location"""
    cfg = DataRequestConfig(
        dataset_class="SomeDatasetClass",
        data_location="https://example.com/public/data.parquet",
        primary_id_field="object_id",
    )
    assert cfg.data_location == "https://example.com/public/data.parquet"


def test_data_request_definition_flat_dict_without_friendly_name_raises():
    """Flat dict with dataset_class at top level (no friendly name) raises ValidationError."""
    with pytest.raises(ValidationError, match="friendly name"):
        DataRequestDefinition(
            {
                "train": {
                    "dataset_class": "HyraxCifarDataset",
                    "primary_id_field": "oid",
                    "data_location": "nowhere",
                }
            }
        )


def test_data_request_definition_bare_instance_without_friendly_name_raises():
    """Bare DataRequestConfig as group value (no friendly name) raises ValidationError."""
    with pytest.raises(ValidationError, match="friendly name"):
        DataRequestDefinition(
            {
                "train": DataRequestConfig(
                    dataset_class="HyraxCifarDataset",
                    primary_id_field="oid",
                    data_location="nowhere",
                )
            }
        )


def test_data_request_definition_collects_known_fields():
    """Collect train/validate/infer fields using required friendly names."""
    definition = DataRequestDefinition(
        {
            "train": {
                "data": {"dataset_class": "TrainDS", "primary_id_field": "id1", "data_location": "nowhere"}
            },
            "validate": {
                "data": {
                    "dataset_class": "ValidateDS",
                    "primary_id_field": "id2",
                    "data_location": "nowhere",
                }
            },
            "infer": {
                "data": {"dataset_class": "InferDS", "primary_id_field": "id3", "data_location": "nowhere"}
            },
        }
    )

    assert isinstance(definition["train"], dict)
    assert isinstance(definition["validate"], dict)
    assert isinstance(definition["infer"], dict)
    assert isinstance(definition["train"]["data"], DataRequestConfig)
    assert isinstance(definition["validate"]["data"], DataRequestConfig)
    assert isinstance(definition["infer"]["data"], DataRequestConfig)


def test_data_request_definition_as_dict_shape():
    """as_dict returns friendly-name keyed dicts without any implicit 'data' wrapper."""
    definition = DataRequestDefinition(
        {
            "train": {
                "data": {
                    "dataset_class": "TrainDS",
                    "fields": ["a"],
                    "primary_id_field": "id1",
                    "data_location": "nowhere",
                }
            },
            "validate": {
                "data": {
                    "dataset_class": "ExtraDS",
                    "primary_id_field": "id2",
                    "data_location": "nowhere",
                }
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
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "fields": ["image"],
                    "primary_id_field": "object_id",
                    "data_location": "nowhere",
                    "dataset_config": {"size": 10, "shape": [1, 2, 3], "seed": 123},
                }
            },
            "infer": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "fields": ["image"],
                    "primary_id_field": "id",
                    "data_location": "somewhere",
                }
            },
        }
    )

    cm._set_config("data_request", definition)

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
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "fields": ["image"],
                "primary_id_field": "object_id",
                "data_location": "nowhere",
            }
        }
    }

    cm._set_config("data_request", valid_dict)

    rendered = cm.config["data_request"]
    assert rendered["train"]["data"]["dataset_class"] == "HyraxRandomDataset"
    assert rendered["train"]["data"]["fields"] == ["image"]
    assert rendered["train"]["data"]["primary_id_field"] == "object_id"


def test_config_manager_set_config_with_invalid_data_accepts_as_is():
    """ConfigManager.set_config accepts invalid data as-is when validation fails."""

    cm = ConfigManager()
    # Invalid: flat dict with dataset_class at top level — no friendly name provided.
    # This fails DataRequestDefinition validation with a "friendly name" error.
    invalid_dict = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "fields": ["image"],
            "data_location": "nowhere",
        }
    }

    # Should not raise - validation error is logged as warning but data is accepted
    cm._set_config("data_request", invalid_dict)

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
            "data": {
                "dataset_class": "HyraxCifarDataset",
                "data_location": "nowhere",
                "primary_id_field": "id",
                "dataset_config": {"use_training_data": True},
            }
        }
    }

    cm._set_config("data_request", valid_dict)

    rendered = cm.config["data_request"]
    assert rendered["train"]["data"]["dataset_class"] == "HyraxCifarDataset"
    assert rendered["train"]["data"]["dataset_config"]["use_training_data"] is True


def test_config_manager_set_config_with_partial_validity():
    """ConfigManager.set_config handles edge case with partially valid data."""

    cm = ConfigManager()
    # Flat dicts (no friendly name) fail DataRequestDefinition validation.
    partially_valid = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "nowhere",
            "primary_id_field": "id",
        },
        "validate": {
            "dataset_class": "HyraxCifarDataset",
            "data_location": "nowhere",
        },
    }

    # Should not raise - validation error is logged as warning but data is accepted
    cm._set_config("data_request", partially_valid)

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


def test_single_config_with_friendly_name_valid():
    """Single config wrapped in a friendly name passes validation."""
    definition = DataRequestDefinition(
        {
            "train": {
                "data": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/dev/null",
                    primary_id_field="object_id",
                )
            }
        }
    )
    assert definition["train"]["data"].primary_id_field == "object_id"


def test_single_config_without_primary_id_fails():
    """Single config without primary_id_field fails validation."""
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            {
                "train": {
                    "data": DataRequestConfig(
                        dataset_class="HyraxRandomDataset",
                        data_location="somewhere",
                    )
                }
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
            "train": {
                "data": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="somewhere",
                    primary_id_field="train_id",
                )
            },
            "validate": {
                "data": DataRequestConfig(
                    dataset_class="HyraxCifarDataset",
                    data_location="somewhere",
                    primary_id_field="validate_id",
                )
            },
            "infer": {
                "data": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="somewhere",
                    primary_id_field="infer_id",
                )
            },
        }
    )
    assert definition["train"]["data"].primary_id_field == "train_id"
    assert definition["validate"]["data"].primary_id_field == "validate_id"
    assert definition["infer"]["data"].primary_id_field == "infer_id"

    # Invalid: train missing primary_id_field
    with pytest.raises(ValidationError) as exc_info:
        DataRequestDefinition(
            {
                "train": {
                    "data": DataRequestConfig(
                        dataset_class="HyraxRandomDataset",
                        data_location="somewhere",
                    )
                },
                "validate": {
                    "data": DataRequestConfig(
                        dataset_class="HyraxCifarDataset",
                        data_location="somewhere",
                        primary_id_field="validate_id",
                    )
                },
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


def test_as_dict_with_single_named_config():
    """as_dict serialises a single named (friendly-name) config correctly."""
    definition = DataRequestDefinition(
        {
            "train": {
                "data": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="somewhere",
                    primary_id_field="id",
                )
            }
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
    # Named configs are serialised without an extra "data" wrapper (see #817).
    assert as_dict["train"]["data_0"]["dataset_class"] == "HyraxRandomDataset"
    assert as_dict["train"]["data_0"]["primary_id_field"] == "id"
    assert as_dict["train"]["data_1"]["dataset_class"] == "HyraxCifarDataset"


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


def test_arbitrary_group_names_accepted():
    """DataRequestDefinition accepts arbitrary group names like 'test' or 'finetune'."""
    definition = DataRequestDefinition(
        {
            "test": {
                "data": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/dev/null",
                    primary_id_field="id",
                )
            },
            "finetune": {
                "data": DataRequestConfig(
                    dataset_class="HyraxCifarDataset",
                    data_location="/dev/null",
                    primary_id_field="id",
                )
            },
        }
    )
    assert definition["test"]["data"].dataset_class == "HyraxRandomDataset"
    assert definition["finetune"]["data"].dataset_class == "HyraxCifarDataset"
    assert "test" in definition
    assert "finetune" in definition


def test_arbitrary_group_name_in_as_dict():
    """as_dict includes arbitrary group names in the output."""
    definition = DataRequestDefinition(
        {
            "test": {
                "data": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/dev/null",
                    primary_id_field="id",
                )
            },
        }
    )
    as_dict = definition.as_dict()
    assert "test" in as_dict
    assert as_dict["test"]["data"]["dataset_class"] == "HyraxRandomDataset"


def test_missing_group_returns_none():
    """Accessing a non-existent group name returns None."""
    definition = DataRequestDefinition(
        {
            "train": {
                "data": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/dev/null",
                    primary_id_field="id",
                )
            },
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
            "train": {
                "data": DataRequestConfig(
                    dataset_class="HyraxRandomDataset",
                    data_location="/dev/null",
                    primary_id_field="id",
                )
            },
            "validate": None,
        }
    )
    assert definition["train"] is not None
    assert definition.root.get("validate") is None
    assert "validate" not in definition


def test_issue_817_single_named_source_no_extra_data_nesting():
    """Regression test for issue #817.

    A single data source under a non-"data" friendly name must not gain an
    extra ``{"data": ...}`` wrapper when the config is round-tripped through
    ``set_config``.  Before the fix, ``config["data_request"]["train"]``
    produced::

        {"friendly_name": {"data": {"dataset_class": ...}}}

    After the fix it correctly returns::

        {"friendly_name": {"dataset_class": ...}}
    """
    cm = ConfigManager()
    data_request = {
        "train": {
            "friendly_name": {
                "dataset_class": "HyraxCifarDataset",
                "data_location": "./data",
                "primary_id_field": "object_id",
            }
        }
    }
    cm._set_config("data_request", data_request)

    train_cfg = cm.config["data_request"]["train"]

    # The friendly name must be present at the top level of the group.
    assert "friendly_name" in train_cfg

    # There must be no spurious "data" wrapper inside the friendly-name entry.
    assert "data" not in train_cfg["friendly_name"], (
        "Extra 'data' nesting was inserted for a single named source (issue #817)."
    )

    # The dataset fields must be directly accessible under the friendly name.
    assert train_cfg["friendly_name"]["dataset_class"] == "HyraxCifarDataset"
    assert train_cfg["friendly_name"]["primary_id_field"] == "object_id"
