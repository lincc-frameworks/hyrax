import numpy as np
import pytest
import torch
from torch import any, from_numpy, isnan, tensor

import hyrax
from hyrax.data_sets.random.hyrax_random_dataset import HyraxRandomDataset
from hyrax.pytorch_ignite import _handle_nans


class RandomNaNDataset(HyraxRandomDataset):
    """Dataset yielding pairs of random numbers. Requires a seed to emulate
    static data on the filesystem between instantiations"""

    def __init__(self, config, data_location):
        super().__init__(config, data_location)

    def __getitem__(self, idx):
        return from_numpy(self.data[idx])


@pytest.fixture(scope="function", params=["RandomNaNDataset", "HyraxRandomDataset"])
def loopback_hyrax_nan(tmp_path_factory, request):
    """This generates a loopback hyrax instance
    which is configured to use the loopback model
    and a simple dataset yielding random numbers
    """
    results_dir = tmp_path_factory.mktemp("loopback_hyrax_nan")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(results_dir)

    h.config["general"]["dev_mode"] = True
    h.config["model_inputs"] = {
        "train": {
            "data": {
                "dataset_class": request.param,
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "infer": {
            "data": {
                "dataset_class": request.param,
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
    }
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]
    h.config["data_set"]["HyraxRandomDataset"]["number_invalid_values"] = 40
    h.config["data_set"]["HyraxRandomDataset"]["invalid_value_type"] = "nan"

    h.config["data_set"]["validate_size"] = 0.2
    h.config["data_set"]["test_size"] = 0.2
    h.config["data_set"]["train_size"] = 0.6

    weights_file = results_dir / "fakeweights"
    with open(weights_file, "a"):
        pass
    h.config["infer"]["model_weights_file"] = str(weights_file)

    dataset = h.prepare()
    return h, dataset["infer"]


def test_nan_handling(loopback_hyrax_nan):
    """
    Test that default nan handling removes nans
    """
    h, dataset = loopback_hyrax_nan
    h.config["data_set"]["nan_mode"] = "quantile"

    inference_results = h.infer()

    if isinstance(dataset[0], dict):
        original_nans = tensor([any(isnan(tensor(item["data"]["image"]))) for item in dataset])
    else:
        original_nans = tensor([any(isnan(item)) for item in dataset])
    assert any(original_nans)

    for result in inference_results:
        assert not any(isnan(result))


def test_nan_handling_zero_values(loopback_hyrax_nan):
    """
    Test that zero nan handling removes nans
    """
    h, dataset = loopback_hyrax_nan
    h.config["data_set"]["nan_mode"] = "zero"

    inference_results = h.infer()

    if isinstance(dataset[0], dict):
        original_nans = tensor([any(isnan(tensor(item["data"]["image"]))) for item in dataset])
    else:
        original_nans = tensor([any(isnan(item)) for item in dataset])
    assert any(original_nans)

    for result in inference_results:
        assert not any(isnan(result))


def test_nan_handling_off(loopback_hyrax_nan):
    """
    Test that when nan handling is off nans appear in output
    """
    h, dataset = loopback_hyrax_nan

    h.config["data_set"]["nan_mode"] = False
    inference_results = h.infer()

    if isinstance(dataset[0], dict):
        original_nans = tensor([any(isnan(tensor(item["data"]["image"]))) for item in dataset])
    else:
        original_nans = tensor([any(isnan(item)) for item in dataset])
    assert any(original_nans)

    result_nans = tensor([any(isnan(item)) for item in inference_results])
    assert any(result_nans)


def test_nan_handling_off_returns_input(loopback_hyrax_nan):
    """Ensure that when nan_mode is False, that the original values passed to
    _handle_nans are returned unchanged."""

    def to_tensor(data_dict):
        data = data_dict.get("data", {})
        if "image" in data and "label" in data:
            image = data["image"]
            label = data["label"]
            return (image, label)

    h, dataset = loopback_hyrax_nan
    h.config["data_set"]["nan_mode"] = False

    sample_data = dataset[0]

    # If the sample data is a dictionary, convert it to a tuple
    if isinstance(sample_data, dict):
        sample_data = to_tensor(sample_data)

    output = _handle_nans(sample_data, h.config)

    # If the sample was a tuple, check all the elements
    if isinstance(sample_data, tuple):
        assert np.all(np.isclose(output[0], sample_data[0], equal_nan=True))
        assert output[1] == sample_data[1]
    else:
        assert np.all(np.isclose(output, sample_data, equal_nan=True))


def test_nan_handling_tuple_with_three_elements(loopback_hyrax_nan):
    """Test that tuples with more than 2 elements are handled correctly.
    This simulates the AppleCider use case where to_tensor returns (metadata, image, labels)."""
    h, dataset = loopback_hyrax_nan

    # Create a test tuple with 3 elements: metadata (non-tensor),
    # image (tensor with NaNs), labels (non-tensor)
    metadata = {"id": 123, "timestamp": "2024-01-01"}
    image_with_nans = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
    labels = "test_label"

    test_tuple = (metadata, image_with_nans, labels)

    # Test with nan_mode = 'zero'
    h.config["data_set"]["nan_mode"] = "zero"
    output = _handle_nans(test_tuple, h.config)

    # Verify we get back a tuple with 3 elements
    assert isinstance(output, tuple)
    assert len(output) == 3

    # Verify metadata is unchanged
    assert output[0] == metadata

    # Verify image had NaNs replaced with zeros
    assert isinstance(output[1], torch.Tensor)
    assert not torch.any(torch.isnan(output[1]))
    assert torch.all(output[1] == torch.tensor([[1.0, 0.0, 3.0], [4.0, 5.0, 0.0]]))

    # Verify labels are unchanged
    assert output[2] == labels


def test_nan_handling_tuple_multiple_tensors(loopback_hyrax_nan):
    """Test that when a tuple contains multiple tensors, NaN handling is applied
    to all of them."""
    h, dataset = loopback_hyrax_nan

    # Create a test tuple with multiple tensors
    tensor1 = torch.tensor([1.0, float("nan"), 3.0])
    tensor2 = torch.tensor([4.0, 5.0, float("nan")])
    label = "label"

    test_tuple = (tensor1, tensor2, label)

    # Test with nan_mode = 'zero'
    h.config["data_set"]["nan_mode"] = "zero"
    output = _handle_nans(test_tuple, h.config)

    # Verify all tensors had NaN handling applied
    assert len(output) == 3
    assert not torch.any(torch.isnan(output[0]))
    assert not torch.any(torch.isnan(output[1]))
    assert output[2] == label

    # Verify the values are correct
    assert torch.all(output[0] == torch.tensor([1.0, 0.0, 3.0]))
    assert torch.all(output[1] == torch.tensor([4.0, 5.0, 0.0]))


def test_nan_handling_tuple_preserves_order(loopback_hyrax_nan):
    """Test that the order of elements in the tuple is preserved."""
    h, dataset = loopback_hyrax_nan

    # Create a test tuple with mixed types in specific order
    elem1 = "first"
    elem2 = torch.tensor([1.0, float("nan")])
    elem3 = {"key": "value"}
    elem4 = torch.tensor([float("nan"), 2.0])
    elem5 = 42

    test_tuple = (elem1, elem2, elem3, elem4, elem5)

    # Test with nan_mode = 'zero'
    h.config["data_set"]["nan_mode"] = "zero"
    output = _handle_nans(test_tuple, h.config)

    # Verify length and order
    assert len(output) == 5
    assert output[0] == elem1
    assert isinstance(output[1], torch.Tensor)
    assert not torch.any(torch.isnan(output[1]))
    assert output[2] == elem3
    assert isinstance(output[3], torch.Tensor)
    assert not torch.any(torch.isnan(output[3]))
    assert output[4] == elem5
