import pytest
from pydantic import ValidationError

from hyrax.config_schemas import ModelInputsConfig, ModelInputsDefinition


def test_model_inputs_config_basic_fields():
    """Basic field round trip."""
    cfg = ModelInputsConfig(
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


def test_model_inputs_config_unwraps_data_key():
    """Support legacy wrapped 'data' key."""
    cfg = ModelInputsConfig(
        data={"dataset_class": "HyraxCifarDataset", "primary_id_field": "oid"}
    )
    assert cfg.dataset_class == "HyraxCifarDataset"
    assert cfg.primary_id_field == "oid"


def test_model_inputs_definition_collects_known_and_extra():
    """Collect train/validate/infer plus extra datasets."""
    definition = ModelInputsDefinition(
        train={"dataset_class": "TrainDS"},
        validate={"dataset_class": "ValidateDS"},
        infer={"dataset_class": "InferDS"},
        custom_split={"dataset_class": "ExtraDS"},
    )

    assert isinstance(definition.train, ModelInputsConfig)
    assert isinstance(definition.validate, ModelInputsConfig)
    assert isinstance(definition.infer, ModelInputsConfig)
    assert "custom_split" in definition.other_datasets
    assert definition.other_datasets["custom_split"].dataset_class == "ExtraDS"


def test_model_inputs_definition_as_dict_shape():
    """as_dict returns nested data blocks."""
    definition = ModelInputsDefinition(
        train={"dataset_class": "TrainDS", "fields": ["a"]},
        custom={"dataset_class": "ExtraDS"},
    )

    as_dict = definition.as_dict()
    assert set(as_dict.keys()) == {"train", "custom"}
    assert as_dict["train"]["data"]["dataset_class"] == "TrainDS"
    assert as_dict["train"]["data"]["fields"] == ["a"]
    assert as_dict["custom"]["data"]["dataset_class"] == "ExtraDS"


def test_model_inputs_definition_rejects_missing_dataset_class():
    """dataset_class is required."""
    with pytest.raises(ValidationError):
        ModelInputsConfig(data_location="/tmp/data")
