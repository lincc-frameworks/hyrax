import numpy as np
import pytest
from torch import any, from_numpy, isnan, tensor
from torch.utils.data import Dataset

import hyrax
from hyrax.data_sets import HyraxDataset
from hyrax.pytorch_ignite import _handle_nans


class RandomNaNDataset(HyraxDataset, Dataset):
    """Dataset yielding pairs of random numbers. Requires a seed to emulate
    static data on the filesystem between instantiations"""

    def __init__(self, config):
        size = config["data_set"]["size"]
        seed = config["data_set"]["seed"]
        rng = np.random.default_rng(seed)
        self.data = rng.random((size, 20), np.float32)

        num_nans = 40
        nan_index_i = rng.integers(0, size, num_nans)
        nan_index_j = rng.integers(0, 20, num_nans)
        for i, j in zip(nan_index_i, nan_index_j):
            self.data[i][j] = np.nan

        # Start our IDs at a random integer between 0 and 100
        id_start = rng.integers(100)
        self.id_list = list(range(id_start, id_start + size))

        super().__init__(config)

    def __getitem__(self, idx):
        return from_numpy(self.data[idx])

    def __len__(self):
        return len(self.data)

    def ids(self):
        """Yield IDs for the dataset"""
        for id_item in self.id_list:
            yield str(id_item)


class RandomNaNTupleDataset(RandomNaNDataset):
    """Subclass of RandomNaNDataset that returns a dictionary. This exercises
    the nan handling logic when working with a supervised dataset."""

    def __getitem__(self, idx):
        value = from_numpy(self.data[idx])
        label = 1
        return {
            "object_id": idx,
            "image": value,
            "label": label,
        }


@pytest.fixture(scope="function", params=["RandomNaNDataset", "RandomNaNTupleDataset"])
def loopback_hyrax_nan(tmp_path_factory, request):
    """This generates a loopback hyrax instance
    which is configured to use the loopback model
    and a simple dataset yielding random numbers
    """
    results_dir = tmp_path_factory.mktemp("loopback_hyrax_nan")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "LoopbackModel"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(results_dir)

    h.config["general"]["dev_mode"] = True
    h.config["data_set"]["name"] = request.param
    h.config["data_set"]["size"] = 20
    h.config["data_set"]["seed"] = 0

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
        original_nans = tensor([any(isnan(item["image"])) for item in dataset])
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
        original_nans = tensor([any(isnan(item["image"])) for item in dataset])
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
        original_nans = tensor([any(isnan(item["image"])) for item in dataset])
    else:
        original_nans = tensor([any(isnan(item)) for item in dataset])
    assert any(original_nans)

    result_nans = tensor([any(isnan(item)) for item in inference_results])
    assert any(result_nans)


def test_nan_handling_off_returns_input(loopback_hyrax_nan):
    """Ensure that when nan_mode is False, that the original values passed to
    _handle_nans are returned unchanged."""

    def to_tensor(data_dict):
        if "image" in data_dict and "label" in data_dict:
            image = data_dict["image"]
            label = data_dict["label"]
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
        assert all(output[0] == sample_data[0])
        assert output[1] == sample_data[1]
    else:
        assert all(output == sample_data)
