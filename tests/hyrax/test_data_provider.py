import builtins
from unittest.mock import patch

import pytest

from hyrax import Hyrax
from hyrax.datasets import HyraxDataset
from hyrax.datasets.data_provider import DataProvider, generate_data_request_from_config


def test_generate_data_request_from_config():
    """Test that an error is raised when no data_request is provided."""

    h = Hyrax()
    config = dict(h.config)
    config.pop("data_request", None)

    config["general"]["data_dir"] = "./data"

    with pytest.raises(RuntimeError, match=r"The \[data_request\] table in the configuration is empty"):
        generate_data_request_from_config(config)


def test_generate_data_request_empty_data_request(caplog):
    """Test that generate_data_request raises an error with a helpful message
    when data_request is empty."""

    h = Hyrax()
    h.config["data_request"] = {}

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError) as execinfo:
            generate_data_request_from_config(h.config)

    error_message = str(execinfo.value)
    assert "The [data_request] table in the configuration is empty." in error_message


def test_data_provider(data_provider):
    """Testing the happy path scenario of creating a DataProvider
    instance with a config that requests two instances of
    `HyraxRandomDataset`.
    """

    dp = data_provider

    assert dp.primary_dataset == "random_0"
    assert dp.primary_dataset_id_field_name == "object_id"

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


def test_validate_request_no_dataset_class(multimodal_config, caplog):
    """Basic test to see that validation works as when no dataset class
    name is provided."""
    h = Hyrax()
    c = multimodal_config
    c["train"]["random_0"].pop("dataset_class", None)
    h.config["data_request"] = c
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError) as execinfo:
            DataProvider(h.config, c["train"])

    assert "does not specify a 'dataset_class'" in str(execinfo.value)
    assert "does not specify a 'dataset_class'" in caplog.text


def test_validate_request_unknown_dataset(multimodal_config, caplog):
    """Basic test to see that validation raises correctly when a nonexistent
    dataset class name is provided."""
    h = Hyrax()
    c = multimodal_config
    c["train"]["random_0"]["dataset_class"] = "NoSuchDataset"
    h.config["data_request"] = c
    with pytest.raises(ValueError) as execinfo:
        DataProvider(h.config, c["train"])

    assert "not found in registry" in str(execinfo.value)


def test_validate_request_bad_field(multimodal_config, caplog):
    """Basic test to see that validation works correctly when a bad field is
    requested."""
    h = Hyrax()
    c = multimodal_config
    c["train"]["random_0"]["fields"] = ["image", "no_such_field"]
    h.config["data_request"] = c
    h.config["data_set"]["preload_cache"] = False  # This reduces warnings on this test
    with caplog.at_level("ERROR"):
        DataProvider(h.config, c["train"])

    assert "No `get_no_such_field` method" in caplog.text


def test_validate_request_dataset_missing_getters(multimodal_config, caplog):
    """Basic test to see that validation works correctly when a dataset is
    missing all getters."""

    h = Hyrax()
    c = multimodal_config
    c["train"]["random_0"].pop("fields", None)
    h.config["data_request"] = c
    h.config["data_set"]["preload_cache"] = False  # This reduces warnings on this test

    # Fake methods to return from `dir`, none of which start with `get_*`.
    fake_methods = ["fake_one", "fake_two", "fake_three"]

    with patch.object(builtins, "dir", return_value=fake_methods):
        with caplog.at_level("ERROR"):
            DataProvider(h.config, c["train"])

    assert "No `get_*` methods were found" in caplog.text


def test_apply_configurations(multimodal_config):
    """Test the static method _apply_configurations to ensure that
    it merges a base config with a dataset-specific config correctly."""

    from hyrax import Hyrax

    h = Hyrax()
    base_config = h.config
    data_request = multimodal_config

    merged_config = DataProvider._apply_configurations(base_config, data_request["train"]["random_0"])

    assert merged_config["data_set"]["HyraxRandomDataset"]["shape"] == [2, 16, 16]
    assert (
        base_config["data_set"]["HyraxRandomDataset"]["seed"]
        == merged_config["data_set"]["HyraxRandomDataset"]["seed"]
    )
    assert merged_config["general"] == base_config["general"]

    merged_config = DataProvider._apply_configurations(base_config, data_request["train"]["random_1"])

    assert base_config["data_set"]["HyraxRandomDataset"]["shape"] != [5, 16, 16]
    assert base_config["data_set"]["HyraxRandomDataset"]["seed"] != 4200
    assert merged_config["data_set"]["HyraxRandomDataset"]["shape"] == [5, 16, 16]
    assert merged_config["data_set"]["HyraxRandomDataset"]["seed"] == 4200
    assert merged_config["general"] == base_config["general"]


def test_apply_configurations_external_dataset():
    """Test that _apply_configurations places external (non-registry)
    dataset_config keys at the top level of the merged config, not
    under 'data_set'."""

    from hyrax import Hyrax

    h = Hyrax()
    base_config = h.config

    dataset_definition = {
        "dataset_class": "SomeExternalDataset",
        "data_location": "/path/to/data",
        "dataset_config": {
            "external_example": {
                "ExternalDataset": {
                    "param1": "value1",
                    "param2": 42,
                },
            },
        },
    }

    merged = DataProvider._apply_configurations(base_config, dataset_definition)

    assert "external_example" in merged
    assert merged["external_example"]["ExternalDataset"]["param1"] == "value1"
    assert merged["external_example"]["ExternalDataset"]["param2"] == 42
    # External keys should NOT appear under data_set
    assert "external_example" not in merged["data_set"]
    # Unrelated sections should be unchanged
    assert merged["general"] == base_config["general"]


def test_apply_configurations_mixed_builtin_and_external():
    """Test that _apply_configurations correctly routes built-in keys
    under 'data_set' and external keys at the top level when both
    are present in the same dataset_config."""

    from hyrax import Hyrax

    h = Hyrax()
    base_config = h.config

    dataset_definition = {
        "dataset_class": "SomeDataset",
        "dataset_config": {
            "HyraxRandomDataset": {
                "shape": [3, 3, 3],
            },
            "ext_lib": {
                "ExtDS": {
                    "foo": "bar",
                },
            },
        },
    }

    merged = DataProvider._apply_configurations(base_config, dataset_definition)

    # Built-in key should be merged under data_set
    assert merged["data_set"]["HyraxRandomDataset"]["shape"] == [3, 3, 3]
    # External key should be at top level
    assert merged["ext_lib"]["ExtDS"]["foo"] == "bar"
    assert "ext_lib" not in merged["data_set"]
    assert merged["general"] == base_config["general"]


def test_apply_configurations_no_dataset_config():
    """Test that _apply_configurations returns the base_config unmodified
    when dataset_definition has no 'dataset_config' key."""

    from hyrax import Hyrax

    h = Hyrax()
    base_config = h.config

    dataset_definition = {
        "dataset_class": "HyraxRandomDataset",
        "data_location": "/path/to/data",
    }

    result = DataProvider._apply_configurations(base_config, dataset_definition)

    # The else branch returns base_config directly (identity)
    assert result is base_config


def test_apply_configurations_empty_dataset_config():
    """Test that _apply_configurations with an empty dataset_config
    returns a config equivalent to the base."""

    from hyrax import Hyrax

    h = Hyrax()
    base_config = h.config

    dataset_definition = {
        "dataset_class": "HyraxRandomDataset",
        "dataset_config": {},
    }

    merged = DataProvider._apply_configurations(base_config, dataset_definition)

    assert merged["data_set"] == base_config["data_set"]
    assert merged["general"] == base_config["general"]


def test_apply_configurations_multiple_builtin_keys():
    """Test that _apply_configurations preserves all built-in keys
    when multiple are present in a single dataset_config."""

    from hyrax import Hyrax

    h = Hyrax()
    base_config = h.config

    dataset_definition = {
        "dataset_class": "SomeDataset",
        "dataset_config": {
            "HyraxRandomDataset": {
                "shape": [7, 7, 7],
            },
            "HyraxCSVDataset": {
                "some_param": "some_value",
            },
        },
    }

    merged = DataProvider._apply_configurations(base_config, dataset_definition)

    # Both built-in keys should survive under data_set
    assert merged["data_set"]["HyraxRandomDataset"]["shape"] == [7, 7, 7]
    assert merged["data_set"]["HyraxCSVDataset"]["some_param"] == "some_value"


def test_primary_dataset(multimodal_config):
    """Test primary dataset selection behavior:
    - uses the dataset with a defined ``primary_id_field`` as primary,
    - raises if no dataset defines ``primary_id_field``,
    - and switches primary when another dataset defines ``primary_id_field``."""

    from hyrax import Hyrax

    h = Hyrax()

    # Base case with `primary_id_field` defined on `random_0`
    data_request = multimodal_config
    h.config["data_request"] = data_request

    dp = DataProvider(h.config, data_request["train"])
    dp.prepare_datasets()

    assert dp.primary_dataset == "random_0"
    assert dp.primary_dataset_id_field_name == "object_id"

    primary_dataset = dp._primary_or_first_dataset()
    assert primary_dataset.data_location == "./in_memory_0"

    # Secondary case with no `primary_id_field` defined
    data_request["train"]["random_0"].pop("primary_id_field", None)
    h.config["data_request"] = data_request

    with pytest.raises(RuntimeError) as execinfo:
        dp = DataProvider(h.config, data_request["train"])

    assert "No Primary Dataset Defined" in str(execinfo.value)
    # Tertiary case with `primary_id_field` defined on `random_1`
    data_request["train"]["random_1"]["primary_id_field"] = "object_id"
    h.config["data_request"] = data_request

    dp = DataProvider(h.config, data_request["train"])
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
                "HyraxRandomDataset": {
                    "shape": [2, 16, 16],
                },
            },
            "primary_id_field": "object_id",
        },
        "random_1": {
            "dataset_class": "HyraxRandomDataset",
            "data_directory": "./in_memory_1",
            "fields": ["image"],
            "dataset_config": {
                "HyraxRandomDataset": {
                    "shape": [5, 16, 16],
                    "seed": 4200,
                },
            },
        },
    }

    h.config["data_request"] = multimodal_config
    dp = DataProvider(h.config, multimodal_config)
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

    # Verify the dataset_config overrides actually took effect
    # (default shape is [2, 5, 5] — these should differ)
    assert sample["random_0"]["image"].shape == (2, 16, 16)
    assert sample["random_1"]["image"].shape == (5, 16, 16)


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

    ids = dp.ids()

    random_0_ids = [
        dp.prepped_datasets["random_0"].get_object_id(idx)
        for idx in range(len(dp.prepped_datasets["random_0"]))
    ]

    random_1_ids = [
        dp.prepped_datasets["random_1"].get_object_id(idx)
        for idx in range(len(dp.prepped_datasets["random_1"]))
    ]

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


def test_primary_id_field_fetched_when_not_in_fields():
    """Test that primary_id_field is fetched on-demand when not in fields list.

    This test validates the fix for the issue where a KeyError occurs when
    primary_id_field is specified but not included in the fields list.
    The fix now fetches the primary_id_field using the dataset getter instead
    of modifying the fields list.
    """
    from hyrax import Hyrax

    h = Hyrax()

    # Configure a dataset where primary_id_field is NOT in the fields list
    # This would previously cause a KeyError in resolve_data
    data_request = {
        "test_dataset": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "./test_data",
            "fields": ["image", "label"],  # Note: "object_id" is NOT included
            "primary_id_field": "object_id",  # But this field is set as primary
            "dataset_config": {
                "HyraxRandomDataset": {
                    "shape": [2, 3, 3],
                    "size": 5,
                    "seed": 42,
                    "provided_labels": ["cat", "dog"],
                    "number_invalid_values": 0,
                    "invalid_value_type": "nan",
                },
            },
        }
    }

    h.config["data_request"] = data_request

    # Create DataProvider
    dp = DataProvider(h.config, data_request)

    # Verify the primary_id_field was NOT added to the fields list
    test_dataset_def = dp.data_request["test_dataset"]
    assert "object_id" not in test_dataset_def["fields"]
    expected_fields = ["image", "label"]
    assert test_dataset_def["fields"] == expected_fields

    # Verify DataProvider was properly configured
    assert dp.primary_dataset == "test_dataset"
    assert dp.primary_dataset_id_field_name == "object_id"

    # This should now work without KeyError - the key test
    # The object_id should be fetched on-demand and added to the top level
    data = dp.resolve_data(0)
    assert "object_id" in data  # Top-level object_id should be present
    assert "test_dataset" in data
    # object_id should NOT be in dataset data since it wasn't requested in fields
    assert "object_id" not in data["test_dataset"]

    # Verify the dataset_config overrides took effect (default shape is [2, 5, 5])
    assert data["test_dataset"]["image"].shape == (2, 3, 3)


def test_primary_id_field_reused_when_already_in_fields():
    """Test that primary_id_field is reused when already in fields list.

    This test validates that when the primary_id_field is already requested
    in the fields list, the resolve_data method reuses that value instead
    of fetching it again.
    """
    from unittest.mock import MagicMock

    from hyrax import Hyrax

    h = Hyrax()

    # Configure a dataset where primary_id_field IS already in the fields list
    data_request = {
        "test_dataset": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "./test_data",
            "fields": ["object_id", "image", "label"],  # object_id already included
            "primary_id_field": "object_id",
            "dataset_config": {
                "HyraxRandomDataset": {
                    "shape": [2, 3, 3],
                    "size": 5,
                    "seed": 42,
                    "provided_labels": ["cat", "dog"],
                    "number_invalid_values": 0,
                    "invalid_value_type": "nan",
                },
            },
        }
    }

    h.config["data_request"] = data_request

    # Create DataProvider - should not duplicate object_id in fields
    dp = DataProvider(h.config, data_request)

    # Verify the fields list is unchanged
    test_dataset_def = dp.data_request["test_dataset"]
    assert test_dataset_def["fields"].count("object_id") == 1
    expected_fields = ["object_id", "image", "label"]
    assert test_dataset_def["fields"] == expected_fields

    # Create a mock for the get_object_id method to track calls
    original_get_object_id = dp.dataset_getters["test_dataset"]["object_id"]
    mock_get_object_id = MagicMock(side_effect=original_get_object_id)
    dp.dataset_getters["test_dataset"]["object_id"] = mock_get_object_id

    # This should work and reuse the existing object_id value
    # The get_object_id should be called exactly once during field resolution,
    # but NOT called again when setting the top-level object_id
    data = dp.resolve_data(0)

    # Verify the get_object_id method was called only once (during field resolution)
    # Since object_id is in the fields list, it gets called once to populate the field,
    # and then the value is reused for the top-level object_id
    assert mock_get_object_id.call_count == 1

    assert "object_id" in data  # Top-level object_id should be present
    assert "test_dataset" in data
    # object_id should be in dataset data since it was requested in fields
    assert "object_id" in data["test_dataset"]

    # The top-level object_id should match the dataset's object_id (reused value)
    assert data["object_id"] == data["test_dataset"]["object_id"]

    # Verify the dataset_config overrides took effect (default shape is [2, 5, 5])
    assert data["test_dataset"]["image"].shape == (2, 3, 3)


def test_collate_function(data_provider):
    """Test that the default collate function in DataProvider
    correctly collates a batch of data samples into a batch dictionary.
    """

    import numpy as np

    dp = data_provider

    # Create a batch of samples
    batch_size = len(dp)
    batch = [dp[i] for i in range(batch_size)]

    # Collate the batch
    collated_batch = dp.collate(batch)

    # Verify the structure of the collated batch
    assert isinstance(collated_batch, dict)
    expected_fields = ["object_id", "image", "label"]
    for field in expected_fields:
        assert field in collated_batch["random_0"]
        assert len(collated_batch["random_0"].keys()) == len(expected_fields)
        assert len(collated_batch["random_0"][field]) == batch_size
        assert isinstance(collated_batch["random_0"][field], np.ndarray)

    expected_fields = ["image"]
    for field in expected_fields:
        assert field in collated_batch["random_1"]
        assert len(collated_batch["random_1"].keys()) == len(expected_fields)
        assert len(collated_batch["random_1"][field]) == batch_size
        assert isinstance(collated_batch["random_1"][field], np.ndarray)

    # assert that the object_id key is a numpy array
    assert isinstance(collated_batch["object_id"], np.ndarray)


def test_finds_custom_collate_function(custom_collate_data_provider):
    """Test that DataProvider correctly identifies datasets
    that have custom collate functions defined.
    """

    dp = custom_collate_data_provider

    assert "random_0" in dp.custom_collate_functions
    assert callable(dp.custom_collate_functions["random_0"])
    assert "random_1" in dp.custom_collate_functions
    assert callable(dp.custom_collate_functions["random_1"])


def test_custom_collate_function_applied(custom_collate_data_provider):
    """Test that DataProvider correctly applies custom collate functions
    for datasets that define them in the DataProvider.collate method.
    """

    import numpy as np

    dp = custom_collate_data_provider

    # Create a batch of samples
    batch_size = len(dp)
    batch = [dp[i] for i in range(batch_size)]

    # Collate the batch
    collated_batch = dp.collate(batch)

    # Verify the structure of the collated batch for random_0
    assert isinstance(collated_batch, dict)

    # Note: expected fields includes "image_mask" which is added by the custom
    # collate function.
    expected_fields = ["object_id", "image", "label", "image_mask"]
    for field in expected_fields:
        assert field in collated_batch["random_0"]
        assert len(collated_batch["random_0"][field]) == batch_size

    # Verify the structure of the collated batch for random_1. Note that "image_mask"
    # is also added by the custom collate function.
    expected_fields = ["image", "image_mask"]
    for field in expected_fields:
        assert field in collated_batch["random_1"]
        assert len(collated_batch["random_1"][field]) == batch_size

    # assert that the object_id key is a numpy array
    assert isinstance(collated_batch["object_id"], np.ndarray)


def test_object_id_is_string():
    """Ensure that the object_id field is returned as a string at the top level of
    the sample, even if the dataset's get_object_id method returns an int. And
    also ensure that the original integer ID is preserved within the dataset-specific
    entry."""

    class IntIDDataset(HyraxDataset):
        """Toy dataset class that returns object_id as an integer."""

        def __init__(self, config, data_location):
            super().__init__(config)

        def __len__(self):
            return 1

        def get_object_id(self, idx):
            """Return the integer object_id for the given index."""
            return idx

    h = Hyrax()
    data_request = {
        "train": {
            "int_id_dataset": {
                "dataset_class": "IntIDDataset",
                "data_location": "./test_data",
                "fields": ["object_id"],
                "primary_id_field": "object_id",
            }
        }
    }
    h.config["data_request"] = data_request

    dp = DataProvider(h.config, data_request["train"])
    dp.prepare_datasets()

    sample = dp[0]
    assert "object_id" in sample
    assert isinstance(sample["object_id"], str)
    assert sample["object_id"] == "0"
    assert sample["int_id_dataset"]["object_id"] == 0


# ---------------------------------------------------------------------------
# Join tests
# ---------------------------------------------------------------------------


class _JoinableDataset(HyraxDataset):
    """Toy dataset whose object IDs and data are fully controllable."""

    def __init__(self, config, data_location, *, ids, values):
        self._ids = list(ids)
        self._values = list(values)
        super().__init__(config)

    def __len__(self):
        return len(self._ids)

    def get_object_id(self, idx):
        return str(self._ids[idx])

    def get_value(self, idx):
        return self._values[idx]


def _make_join_provider(primary_ids, primary_vals, secondary_ids, secondary_vals):
    """Helper: build a DataProvider with a primary and a joined secondary."""
    from hyrax import Hyrax

    h = Hyrax()

    # Stash data on the class so __init__ can pick it up.
    _JoinableDataset._pending_primary = (primary_ids, primary_vals)
    _JoinableDataset._pending_secondary = (secondary_ids, secondary_vals)

    # We need two distinct classes so they get separate instances.
    class PrimaryDS(_JoinableDataset):
        def __init__(self, config, data_location):
            ids, vals = _JoinableDataset._pending_primary
            super().__init__(config, data_location, ids=ids, values=vals)

    class SecondaryDS(_JoinableDataset):
        def __init__(self, config, data_location):
            ids, vals = _JoinableDataset._pending_secondary
            super().__init__(config, data_location, ids=ids, values=vals)

    request = {
        "primary": {
            "dataset_class": "PrimaryDS",
            "data_location": "./mem_primary",
            "fields": ["object_id", "value"],
            "primary_id_field": "object_id",
        },
        "secondary": {
            "dataset_class": "SecondaryDS",
            "data_location": "./mem_secondary",
            "fields": ["object_id", "value"],
            "join_field": "object_id",
        },
    }

    h.config["data_request"] = {"train": request}
    return DataProvider(h.config, request)


def test_join_basic_matching():
    """Joined datasets with overlapping IDs return correctly paired data."""
    dp = _make_join_provider(
        primary_ids=["A", "B", "C"],
        primary_vals=[10, 20, 30],
        secondary_ids=["C", "A", "B"],  # same IDs, different order
        secondary_vals=[300, 100, 200],
    )

    assert len(dp) == 3

    # Collect all resolved pairs
    pairs = {}
    for i in range(len(dp)):
        sample = dp[i]
        oid = sample["object_id"]
        pairs[oid] = (sample["primary"]["value"], sample["secondary"]["value"])

    assert pairs["A"] == (10, 100)
    assert pairs["B"] == (20, 200)
    assert pairs["C"] == (30, 300)


def test_join_left_outer_none_for_missing():
    """Unmatched primary items get None for the joined secondary (left outer join)."""
    dp = _make_join_provider(
        primary_ids=["A", "B", "C", "D"],
        primary_vals=[1, 2, 3, 4],
        secondary_ids=["B", "D", "E"],
        secondary_vals=[20, 40, 50],
    )

    # All primary items are present.
    assert len(dp) == 4

    ids_seen = {dp.get_object_id(i) for i in range(len(dp))}
    assert ids_seen == {"A", "B", "C", "D"}

    # B and D have secondary data; A and C are None.
    sample_a = dp[0]
    assert sample_a["object_id"] == "A"
    assert sample_a["secondary"] is None

    sample_b = dp[1]
    assert sample_b["object_id"] == "B"
    assert sample_b["secondary"]["value"] == 20


def test_join_preserves_primary_order():
    """All primary items appear in their original order."""
    dp = _make_join_provider(
        primary_ids=["X", "Y", "Z", "W"],
        primary_vals=[1, 2, 3, 4],
        secondary_ids=["Z", "X"],
        secondary_vals=[30, 10],
    )

    assert len(dp) == 4
    assert dp.get_object_id(0) == "X"
    assert dp.get_object_id(1) == "Y"
    assert dp.get_object_id(2) == "Z"
    assert dp.get_object_id(3) == "W"

    # Y and W have no match
    assert dp[1]["secondary"] is None
    assert dp[3]["secondary"] is None
    # X and Z have matches
    assert dp[0]["secondary"]["value"] == 10
    assert dp[2]["secondary"]["value"] == 30


def test_join_ids_method():
    """DataProvider.ids() returns ALL primary IDs (left outer join)."""
    dp = _make_join_provider(
        primary_ids=["A", "B", "C"],
        primary_vals=[1, 2, 3],
        secondary_ids=["B", "C"],
        secondary_vals=[20, 30],
    )

    assert dp.ids() == ["A", "B", "C"]


def test_join_zero_overlap_allowed():
    """Zero overlap between primary and secondary is allowed (all None)."""
    dp = _make_join_provider(
        primary_ids=["A", "B"],
        primary_vals=[1, 2],
        secondary_ids=["C", "D"],
        secondary_vals=[3, 4],
    )

    assert len(dp) == 2
    assert dp[0]["secondary"] is None
    assert dp[1]["secondary"] is None


def test_join_collate():
    """Collation works correctly with fully-matched joined datasets."""
    import numpy as np

    dp = _make_join_provider(
        primary_ids=["A", "B"],
        primary_vals=[10, 20],
        secondary_ids=["B", "A"],
        secondary_vals=[200, 100],
    )

    batch = [dp[i] for i in range(len(dp))]
    collated = dp.collate(batch)

    assert "primary" in collated
    assert "secondary" in collated
    assert "object_id" in collated
    assert isinstance(collated["object_id"], np.ndarray)
    assert len(collated["object_id"]) == 2
    # No __matched mask when all items match
    assert "secondary__matched" not in collated


def test_join_collate_with_missing():
    """Collation adds __matched mask and passes through None for unmatched items."""
    import numpy as np

    dp = _make_join_provider(
        primary_ids=["A", "B", "C"],
        primary_vals=[1, 2, 3],
        secondary_ids=["B"],
        secondary_vals=[20],
    )

    batch = [dp[i] for i in range(len(dp))]
    collated = dp.collate(batch)

    # Primary is always fully present
    assert "primary" in collated
    assert len(collated["primary"]["value"]) == 3

    # Secondary should NOT have collated data for all 3 — only the 1 match
    # was aggregated.  The __matched mask tells the consumer which positions
    # have real data.
    assert "secondary__matched" in collated
    mask = collated["secondary__matched"]
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert list(mask) == [False, True, False]

    # The secondary dict should only contain the 1 matched item.
    assert "secondary" in collated
    assert len(collated["secondary"]["value"]) == 1


def test_join_field_mutually_exclusive_with_primary_id_field():
    """join_field and primary_id_field cannot both be set on the same dataset."""
    from hyrax.config_schemas.data_request import DataRequestConfig

    with pytest.raises(ValueError, match="mutually exclusive"):
        DataRequestConfig(
            dataset_class="Foo",
            data_location="./x",
            primary_id_field="object_id",
            join_field="object_id",
        )


def test_no_join_field_preserves_existing_behavior(data_provider):
    """When no join_field is set, DataProvider behaves exactly as before."""
    dp = data_provider

    assert dp._join_maps == {}

    # Verify normal index-aligned access works
    sample = dp[0]
    assert "random_0" in sample
    assert "random_1" in sample
    assert "object_id" in sample


def test_join_cache_roundtrip(tmp_path):
    """Verify that join maps are persisted and reloaded from disk."""
    from hyrax.datasets.data_provider import _load_join_cache, _save_join_cache

    ids = ["X", "Y", "Z"]

    def getter(idx):
        return ids[idx]

    reverse_map = {str(k): i for i, k in enumerate(ids)}

    # Save to tmp_path (which exists as a directory).
    data_location = str(tmp_path / "fake_data.csv")

    _save_join_cache(data_location, len(ids), getter, reverse_map)

    # Reload should succeed and return the same map.
    loaded = _load_join_cache(data_location, len(ids), getter)
    assert loaded is not None
    assert loaded == reverse_map


def test_join_cache_invalidated_on_length_change(tmp_path):
    """Cache is invalidated when the dataset length changes."""
    from hyrax.datasets.data_provider import _load_join_cache, _save_join_cache

    ids = ["A", "B", "C"]

    def getter(idx):
        return ids[idx]

    reverse_map = {"A": 0, "B": 1, "C": 2}
    data_location = str(tmp_path / "data.csv")

    _save_join_cache(data_location, 3, getter, reverse_map)

    # Attempting to load with a different length should miss.
    loaded = _load_join_cache(data_location, 4, getter)
    assert loaded is None


def test_join_cache_invalidated_on_key_change(tmp_path):
    """Cache is invalidated when sampled key values change."""
    from hyrax.datasets.data_provider import _load_join_cache, _save_join_cache

    ids_v1 = ["A", "B", "C"]
    ids_v2 = ["A", "B", "D"]  # last key changed

    def getter_v1(idx):
        return ids_v1[idx]

    def getter_v2(idx):
        return ids_v2[idx]

    reverse_map = {"A": 0, "B": 1, "C": 2}
    data_location = str(tmp_path / "data.csv")

    _save_join_cache(data_location, 3, getter_v1, reverse_map)

    # Load with changed keys should miss.
    loaded = _load_join_cache(data_location, 3, getter_v2)
    assert loaded is None


def test_join_parallel_build():
    """Join maps for multiple secondaries are built (potentially in parallel)."""
    from hyrax import Hyrax

    class PrimaryMulti(_JoinableDataset):
        def __init__(self, config, data_location):
            super().__init__(config, data_location, ids=["A", "B", "C"], values=[1, 2, 3])

    class SecondaryMultiA(_JoinableDataset):
        def __init__(self, config, data_location):
            super().__init__(config, data_location, ids=["C", "B", "A"], values=[30, 20, 10])

    class SecondaryMultiB(_JoinableDataset):
        def __init__(self, config, data_location):
            super().__init__(config, data_location, ids=["A", "C"], values=[100, 300])

    h = Hyrax()
    request = {
        "primary": {
            "dataset_class": "PrimaryMulti",
            "data_location": "./mem",
            "fields": ["object_id", "value"],
            "primary_id_field": "object_id",
        },
        "sec_a": {
            "dataset_class": "SecondaryMultiA",
            "data_location": "./mem_a",
            "fields": ["value"],
            "join_field": "object_id",
        },
        "sec_b": {
            "dataset_class": "SecondaryMultiB",
            "data_location": "./mem_b",
            "fields": ["value"],
            "join_field": "object_id",
        },
    }

    h.config["data_request"] = {"train": request}
    dp = DataProvider(h.config, request)

    # Left outer join: all 3 primary items are present
    assert len(dp) == 3

    for i in range(len(dp)):
        sample = dp[i]
        oid = sample["object_id"]
        if oid == "A":
            assert sample["primary"]["value"] == 1
            assert sample["sec_a"]["value"] == 10
            assert sample["sec_b"]["value"] == 100
        elif oid == "B":
            assert sample["primary"]["value"] == 2
            assert sample["sec_a"]["value"] == 20
            assert sample["sec_b"] is None  # B not in sec_b
        else:
            assert oid == "C"
            assert sample["primary"]["value"] == 3
            assert sample["sec_a"]["value"] == 30
            assert sample["sec_b"]["value"] == 300
