import numpy as np
import pytest
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
        "data": {
            "dataset_class": request.param,
            "data_location": str(tmp_path_factory.mktemp("data")),
            "primary_id_field": "object_id",
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
    return h, dataset


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
            image = tensor(data["image"])
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
