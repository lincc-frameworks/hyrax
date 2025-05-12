import numpy as np
import pytest
import torch.nn as nn
from torch import from_numpy
from torch.utils.data import Dataset, IterableDataset

import hyrax
from hyrax.data_sets import HyraxDataset
from hyrax.models import hyrax_model


@hyrax_model
class LoopbackModel(nn.Module):
    """Simple model for testing which returns its own input"""

    def __init__(self, config, shape):
        super().__init__()
        # This is created so the optimizier can find at least one weight
        self.unused_module = nn.Conv2d(1, 1, kernel_size=1, stride=0, padding=0)
        self.config = config

    def forward(self, x):
        """We simply return our input"""
        return x

    def train_step(self, batch):
        """Training is a noop"""
        return {"loss": 0.0}


class RandomDataset(HyraxDataset, Dataset):
    """Dataset yielding pairs of random numbers. Requires a seed to emulate
    static data on the filesystem between instantiations"""

    def __init__(self, config):
        size = config["data_set"]["size"]
        seed = config["data_set"]["seed"]
        rng = np.random.default_rng(seed)
        self.data = rng.random((size, 2), np.float32)

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


class RandomIterableDataset(RandomDataset, IterableDataset):
    """Iterable version of RandomDataset"""

    def __iter__(self):
        for item in self.data:
            yield from_numpy(item)


@pytest.fixture(scope="function", params=["RandomDataset", "RandomIterableDataset"])
def loopback_hyrax(tmp_path_factory, request):
    """This generates a loopback hyrax instance
    which is configured to use the loopback model
    and a simple dataset yielding random numbers
    """
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "LoopbackModel"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(tmp_path_factory.mktemp(f"loopback_hyrax_{request.param}"))

    h.config["general"]["dev_mode"] = True
    h.config["data_set"]["name"] = request.param
    h.config["data_set"]["size"] = 20
    h.config["data_set"]["seed"] = 0

    h.config["data_set"]["validate_size"] = 0.2
    h.config["data_set"]["test_size"] = 0.2
    h.config["data_set"]["train_size"] = 0.6

    dataset = h.prepare()
    h.train()

    return h, dataset


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("split", ["test", "train", "validate", None])
def test_infer_order(loopback_hyrax, split, shuffle):
    """Test that the order of data run through infer
    is correct in the presence of several splits
    """
    h, dataset = loopback_hyrax
    h.config["infer"]["split"] = split if split is not None else False
    h.config["data_loader"]["shuffle"] = shuffle

    inference_results = h.infer()
    inference_result_ids = list(inference_results.ids())
    original_dataset_ids = list(dataset.ids())

    for idx, result_id in enumerate(inference_result_ids):
        dataset_idx = None
        for i, orig_id in enumerate(original_dataset_ids):
            if orig_id == result_id:
                dataset_idx = i
                break
        else:
            raise AssertionError("Failed to find a corresponding ID")

        print(f"orig idx: {dataset_idx}, infer idx: {idx}")
        print(f"orig data: {dataset[dataset_idx]}, infer data: {inference_results[idx]}")
        assert all(np.isclose(dataset[dataset_idx], inference_results[idx]))
