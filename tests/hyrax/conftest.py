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
        from functools import partial

        super().__init__()
        # This is created so the optimizier can find at least one weight
        self.unused_module = nn.Conv2d(1, 1, kernel_size=1, stride=0, padding=0)
        self.config = config

        def load(self, weight_file):
            """Load Weights, we have no weights so we do nothing"""
            pass

        # We override this way rather than defining a method because
        # Torch has some __init__ related cleverness which stomps our
        # load definition when performed in the usual fashion.
        self.load = partial(load, self)

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
        dim_1_length = config["data_set"]["dimension_1_length"]

        dim_2_length = (
            config["data_set"]["dimension_2_length"] if config["data_set"]["dimension_2_length"] else 0
        )

        seed = config["data_set"]["seed"]
        rng = np.random.default_rng(seed)

        if dim_2_length > 0:
            self.data = rng.random((size, dim_1_length, dim_2_length), np.float32)
        else:
            self.data = rng.random((size, dim_1_length), np.float32)

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
    results_dir = tmp_path_factory.mktemp(f"loopback_hyrax_{request.param}")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "LoopbackModel"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(results_dir)

    h.config["general"]["dev_mode"] = True
    h.config["data_set"]["name"] = request.param
    h.config["data_set"]["size"] = 20
    h.config["data_set"]["seed"] = 0
    h.config["data_set"]["dimension_1_length"] = 2
    h.config["data_set"]["dimension_2_length"] = 3

    h.config["data_set"]["validate_size"] = 0.2
    h.config["data_set"]["test_size"] = 0.2
    h.config["data_set"]["train_size"] = 0.6

    weights_file = results_dir / "fakeweights"
    with open(weights_file, "a"):
        pass
    h.config["infer"]["model_weights_file"] = str(weights_file)

    dataset = h.prepare()
    return h, dataset


@pytest.fixture(scope="function")
def loopback_inferred_hyrax(loopback_hyrax):
    """This generates a loopback hyrax instance which is configured to use the
    loopback model and a simple dataset yielding random numbers. It includes a call
    to hyrax.infer which will produce the output consumed by vdb_index or umap."""

    h, dataset = loopback_hyrax
    inference_results = h.infer()

    return h, dataset, inference_results
