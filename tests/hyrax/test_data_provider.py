import builtins
from unittest.mock import patch

import pytest

from hyrax import Hyrax
from hyrax.data_sets.data_provider import DataProvider, generate_data_request_from_config


def test_generate_data_request_from_config():
    """Test that we support generating a data request dictionary
    outside of the `model_inputs` table."""

    h = Hyrax()
    config = dict(h.config)
    config.pop("model_inputs", None)

    config["data_set"]["name"] = "HyraxRandomDataset"
    config["general"]["data_dir"] = "./data"

    ret_val = generate_data_request_from_config(config)

    assert "data" in ret_val
    assert "dataset_class" in ret_val["data"]
    assert ret_val["data"]["dataset_class"] == "HyraxRandomDataset"
    assert "data_location" in ret_val["data"]
    assert ret_val["data"]["data_location"] == "./data"


def test_generate_data_request_passes_model_inputs():
    """Test that generate_data_request passes the model_inputs
    dict from the config, unchanged."""

    h = Hyrax()
    h.config["model_inputs"] = {
        "a": "foo",
        "b": {"c": "bar"},
    }

    ret_val = generate_data_request_from_config(h.config)

    assert "a" in ret_val
    assert ret_val["a"] == "foo"
    assert "b" in ret_val
    assert isinstance(ret_val["b"], dict)
    assert ret_val["b"] == {"c": "bar"}


def test_data_provider(data_provider):
    """Testing the happy path scenario of creating a DataProvider
    instance with a config that requests two instances of
    `HyraxRandomDataset`.
    """

    dp = data_provider

    assert dp.primary_dataset == "random_0"
    assert dp.primary_dataset_id_field_name == "object_id"

    assert dp.is_iterable() is False
    assert dp.is_map() is True

    # There should be 2 prepared datasets
    assert len(dp.prepped_datasets) == 2
    assert "random_0" in dp.prepped_datasets
    assert "random_1" in dp.prepped_datasets

    # There should be 2 dataset_getters dicts with subdicts of different sizes
    assert len(dp.dataset_getters) == 2
    assert len(dp.dataset_getters["random_0"]) == 5
    assert len(dp.dataset_getters["random_1"]) == 5

    data_request = dp.data_request
    for friendly_name in data_request:
        for field in data_request[friendly_name]["fields"]:
            assert field in dp.dataset_getters[friendly_name]

    data_request = dp.data_request
    for friendly_name in data_request:
        assert len(dp.all_metadata_fields[friendly_name]) == 3
        for metadata_field in dp.all_metadata_fields[friendly_name]:
            assert friendly_name in metadata_field


def test_validate_request(multimodal_config):
    """Basic test to see that validation works as expected in the base
    case."""

    c = multimodal_config
    DataProvider.validate_request(c)


def test_validate_request_no_dataset_class(multimodal_config, caplog):
    """Basic test to see that validation works as when no dataset class
    name is provided."""

    c = multimodal_config
    c["random_0"].pop("dataset_class", None)
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError) as execinfo:
            DataProvider.validate_request(c)

    assert "failed" in str(execinfo.value)
    assert "does not specify a 'dataset_class'" in caplog.text


def test_validate_request_unknown_dataset(multimodal_config, caplog):
    """Basic test to see that validation raises correctly when a nonexistent
    dataset class name is provided."""

    c = multimodal_config
    c["random_0"]["dataset_class"] = "NoSuchDataset"
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError) as execinfo:
            DataProvider.validate_request(c)

    assert "failed" in str(execinfo.value)
    assert "Unable to locate dataset" in caplog.text


def test_validate_request_iterable_dataset(multimodal_config, caplog):
    """Basic test to see that validation works correctly when an iterable dataset
    is requested"""

    c = multimodal_config
    c["random_0"]["dataset_class"] = "HyraxRandomIterableDataset"
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError) as execinfo:
            DataProvider.validate_request(c)

    assert "failed" in str(execinfo.value)
    assert "is an iterable-style dataset" in caplog.text


def test_validate_request_bad_field(multimodal_config, caplog):
    """Basic test to see that validation works correctly when a bad field is
    requested."""
    h = Hyrax()
    c = multimodal_config
    c["random_0"]["fields"] = ["image", "no_such_field"]
    h.config["model_inputs"] = c
    with caplog.at_level("ERROR"):
        DataProvider(h.config)

    assert "No `get_no_such_field` method" in caplog.text


def test_validate_request_dataset_missing_getters(multimodal_config, caplog):
    """Basic test to see that validation works correctly when a dataset is
    missing all getters."""

    h = Hyrax()
    c = multimodal_config
    c["random_0"].pop("fields", None)
    h.config["model_inputs"] = c

    # Fake methods to return from `dir`, none of which start with `get_*`.
    fake_methods = ["fake_one", "fake_two", "fake_three"]

    with patch.object(builtins, "dir", return_value=fake_methods):
        with caplog.at_level("ERROR"):
            DataProvider(h.config)

    assert "No `get_*` methods were found" in caplog.text


def test_apply_configurations(multimodal_config):
    """Test the static method _apply_configurations to ensure that
    it merges a base config with a dataset-specific config correctly."""

    from hyrax import Hyrax

    h = Hyrax()
    base_config = h.config
    model_inputs = multimodal_config

    merged_config = DataProvider._apply_configurations(base_config, model_inputs["random_0"])

    assert merged_config["data_set"]["HyraxRandomDataset"]["shape"] == [2, 16, 16]
    assert (
        base_config["data_set"]["HyraxRandomDataset"]["seed"]
        == merged_config["data_set"]["HyraxRandomDataset"]["seed"]
    )
    assert merged_config["general"] == base_config["general"]

    merged_config = DataProvider._apply_configurations(base_config, model_inputs["random_1"])

    assert base_config["data_set"]["HyraxRandomDataset"]["shape"] != [5, 16, 16]
    assert base_config["data_set"]["HyraxRandomDataset"]["seed"] != 4200
    assert merged_config["data_set"]["HyraxRandomDataset"]["shape"] == [5, 16, 16]
    assert merged_config["data_set"]["HyraxRandomDataset"]["seed"] == 4200
    assert merged_config["general"] == base_config["general"]


def test_primary_or_first_dataset(multimodal_config):
    """Test that if no primary dataset is specified, the first dataset
    in the config is returned."""

    from hyrax import Hyrax

    h = Hyrax()

    # Base case with `primary_id_field` defined on `random_0`
    model_inputs = multimodal_config
    h.config["model_inputs"] = model_inputs

    dp = DataProvider(h.config)
    dp.prepare_datasets()

    assert dp.primary_dataset == "random_0"
    assert dp.primary_dataset_id_field_name == "object_id"

    primary_dataset = dp._primary_or_first_dataset()
    assert primary_dataset.data_location == "./in_memory_0"

    # Secondary case with no `primary_id_field` defined
    model_inputs["random_0"].pop("primary_id_field", None)
    h.config["model_inputs"] = model_inputs

    dp = DataProvider(h.config)
    dp.prepare_datasets()

    assert dp.primary_dataset is None
    assert dp.primary_dataset_id_field_name is None

    primary_dataset = dp._primary_or_first_dataset()
    assert primary_dataset.data_location == "./in_memory_0"

    # Tertiary case with `primary_id_field` defined on `random_1`
    model_inputs["random_1"]["primary_id_field"] = "object_id"
    h.config["model_inputs"] = model_inputs

    dp = DataProvider(h.config)
    dp.prepare_datasets()

    assert dp.primary_dataset == "random_1"
    assert dp.primary_dataset_id_field_name == "object_id"

    primary_dataset = dp._primary_or_first_dataset()
    assert primary_dataset.data_location == "./in_memory_1"


def test_metadata_fields(data_provider):
    """Test that the calling metadata_fields returns the expected
    fields with the expected structure."""

    dp = data_provider
    dp.prepare_datasets()

    all_metadata_fields = dp.all_metadata_fields

    assert "random_0" in all_metadata_fields
    assert "random_1" in all_metadata_fields

    assert len(all_metadata_fields["random_0"]) == 3
    assert len(all_metadata_fields["random_1"]) == 3

    all_fields = dp.metadata_fields()

    assert isinstance(all_fields, list)
    assert "object_id" in all_fields

    expected_metadata_fields = ["object_id", "meta_field_1", "meta_field_2"]
    for field in expected_metadata_fields:
        assert field + "_random_0" in all_fields
        assert field + "_random_1" in all_fields


def test_metadata_fields_with_friendly_name(data_provider):
    """Test that the calling metadata_fields returns the expected
    fields with the expected structure."""

    dp = data_provider
    dp.prepare_datasets()

    all_fields = dp.metadata_fields("random_0")

    assert isinstance(all_fields, list)
    assert "object_id" in all_fields

    expected_metadata_fields = ["object_id", "meta_field_1", "meta_field_2"]
    for field in expected_metadata_fields:
        assert field in all_fields


def test_sample_data():
    """Test that sample_data returns a dictionary with the expected
    structure.

    We don't use the test fixture here so that this can be a little more
    flexible and self-contained.


    The expected result structure is:
    {
        'random_0': {
            'object_id': <int>,
            'image': array(...),
            'label': 'cat'
        },
        'random_1': {
            'image': array(...)
        },
        'object_id': <int>
    }
    """
    from hyrax import Hyrax

    h = Hyrax()

    multimodal_config = {
        "random_0": {
            "dataset_class": "HyraxRandomDataset",
            "data_directory": "./in_memory_0",
            "fields": ["object_id", "image", "label"],
            "dataset_config": {
                "shape": [2, 16, 16],
            },
            "primary_id_field": "object_id",
        },
        "random_1": {
            "dataset_class": "HyraxRandomDataset",
            "data_directory": "./in_memory_1",
            "fields": ["image"],
            "dataset_config": {
                "shape": [5, 16, 16],
                "seed": 4200,
            },
        },
    }

    h.config["model_inputs"] = multimodal_config
    dp = DataProvider(h.config)
    dp.prepare_datasets()

    sample = dp.sample_data()

    assert isinstance(sample, dict)
    assert "random_0" in sample
    assert "random_1" in sample
    assert "object_id" in sample
    assert len(sample) == 3
    assert isinstance(sample["random_0"], dict)
    assert isinstance(sample["random_1"], dict)
    assert len(sample["random_0"]) == 3
    assert len(sample["random_1"]) == 1

    for friendly_name in ["random_0", "random_1"]:
        dataset_sample = sample[friendly_name]
        assert "image" in dataset_sample

        if friendly_name == "random_0":
            assert "object_id" in dataset_sample
            assert "label" in dataset_sample


def test_data_provider_get_item(data_provider):
    """Basic test to ensure that different index values return different data."""
    dp = data_provider
    dp.prepare_datasets()

    sample_0a = dp[0]
    sample_0b = dp.resolve_data(0)
    sample_1a = dp[1]
    sample_1b = dp.resolve_data(1)

    assert isinstance(sample_0a, dict)
    assert isinstance(sample_1a, dict)
    assert sample_0a["random_0"]["image"][0][0][0] != sample_1a["random_0"]["image"][0][0][0]
    assert sample_0a["random_0"]["image"][0][0][0] == sample_0b["random_0"]["image"][0][0][0]

    assert isinstance(sample_0b, dict)
    assert isinstance(sample_1b, dict)
    assert sample_0b["random_0"]["image"][0][0][0] != sample_1b["random_0"]["image"][0][0][0]
    assert sample_1a["random_0"]["image"][0][0][0] == sample_1b["random_0"]["image"][0][0][0]

    assert sample_0a["random_0"]["object_id"] == sample_0b["random_0"]["object_id"]
    assert sample_1a["random_0"]["object_id"] == sample_1b["random_0"]["object_id"]
    assert "object_id" in sample_0a
    assert "object_id" in sample_1a
    assert "object_id" in sample_0b
    assert "object_id" in sample_1b


def test_data_provider_returns_length(data_provider):
    """Basic test to ensure that __len__ returns the expected value.
    The length returned from DataProvider is the length of the primary
    dataset or, if no primary dataset is specified.

    We have specified random_0 to be the primary dataset in conftest.
    """

    dp = data_provider
    dp.prepare_datasets()

    length = len(dp)

    random_0_length = len(dp.prepped_datasets["random_0"])
    assert isinstance(length, int)
    assert length == random_0_length


def test_data_provider_ids(data_provider):
    """Basic test to ensure that ids() returns the expected value.
    The ids returned from DataProvider are the ids of the primary
    dataset or, if no primary dataset is specified, the first dataset.

    We have specified random_0 to be the primary dataset in conftest.
    """

    dp = data_provider
    dp.prepare_datasets()

    ids = list(dp.ids())

    random_0_ids = list(dp.prepped_datasets["random_0"].ids())
    random_1_ids = list(dp.prepped_datasets["random_1"].ids())

    assert len(ids) == len(random_0_ids)
    assert all(i == j for i, j in zip(ids, random_0_ids))
    assert all(i != j for i, j in zip(ids, random_1_ids))


def test_data_provider_returns_metadata(data_provider):
    """Basic test to ensure that metadata() returns the expected value.
    The metadata returned from DataProvider are the metadata of the primary
    dataset or, if no primary dataset is specified, the first dataset.

    We have specified random_0 to be the primary dataset in conftest.
    """

    dp = data_provider
    dp.prepare_datasets()

    metadata = dp.metadata()
    assert len(metadata) == 0

    metadata = dp.metadata(idxs=None, fields=[])
    assert len(metadata) == 0

    metadata = dp.metadata(idxs=[], fields=[])
    assert len(metadata) == 0

    metadata = dp.metadata(idxs=[], fields=["meta_field_1_random_0"])
    assert len(metadata) == 0
    assert len(metadata.dtype.names) == 1
    assert "meta_field_1_random_0" in metadata.dtype.names[0]

    metadata = dp.metadata(idxs=[5], fields=["meta_field_1_random_0"])
    assert len(metadata) == 1
    assert len(metadata.dtype.names) == 1
    assert "meta_field_1_random_0" in metadata.dtype.names[0]

    metadata = dp.metadata(idxs=[5, 97], fields=["meta_field_1_random_0"])
    assert len(metadata) == 2
    assert len(metadata.dtype.names) == 1
    assert "meta_field_1_random_0" in metadata.dtype.names

    metadata = dp.metadata(idxs=[5, 97], fields=["meta_field_1_random_0", "meta_field_2_random_1"])
    assert len(metadata) == 2
    assert len(metadata.dtype.names) == 2
    assert "meta_field_1_random_0" in metadata.dtype.names
    assert "meta_field_2_random_1" in metadata.dtype.names
