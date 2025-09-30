import logging
import sys

import numpy as np
import pytest
from astropy.table import Table
from torch import from_numpy
from torch.utils.data import Dataset, IterableDataset

import hyrax
from hyrax.data_sets import HyraxDataset
from hyrax.data_sets.data_provider import DataProvider

logger = logging.getLogger(__name__)


def pytest_configure(config):
    """
    Global test configuration. We:
    1) Disable ConfigManager from slurping up files from the working directory to enable test reproducibility
       across different developer machines and CI.

    2) Set an unlimited number of open files per process on OSX. OSX's default per-process file limit is 256
       Because we use temporary files during many of our tests, it's easy to go over this limit.
    """
    hyrax.config_utils.ConfigManager._called_from_test = True

    if sys.platform == "darwin":
        import resource

        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ValueError as e:
            msg = "Attempted to raise open file limit, and failed. Tests may not work.\n"
            msg += f"See error below when trying to raise open file limit: \n {e}"
            raise RuntimeError(msg) from e


class RandomDataset(HyraxDataset, Dataset):
    """Dataset yielding pairs of random numbers. Requires a seed to emulate
    static data on the filesystem between instantiations"""

    def __init__(self, config, data_directory):
        size = config["data_set"]["size"]

        dim_1_length = 2
        if "dimension_1_length" in config["data_set"]:
            dim_1_length = config["data_set"]["dimension_1_length"]

        dim_2_length = 0
        if "dimension_2_length" in config["data_set"]:
            dim_2_length = config["data_set"]["dimension_2_length"]

        seed = config["data_set"]["seed"]
        rng = np.random.default_rng(seed)

        print(f"Initialized dataset with dim 1: {dim_1_length}, dim 2: {dim_2_length}")

        if dim_2_length > 0:
            self.data = rng.random((size, dim_1_length, dim_2_length), np.float32)
        else:
            self.data = rng.random((size, dim_1_length), np.float32)

        # Start our IDs at a random integer between 0 and 100
        id_start = rng.integers(100)
        self.id_list = list(range(id_start, id_start + size))

        metadata_table = Table({"object_id": np.array(list(self.ids()))})

        super().__init__(config, metadata_table)

    def __getitem__(self, idx):
        return from_numpy(self.data[idx])

    def __len__(self):
        return len(self.data)

    def ids(self):
        """Yield IDs for the dataset"""
        for id_item in self.id_list:
            yield str(id_item)

    def get_ids(self, idx):
        """Returns the ids given an index."""
        return self.id_list[idx]


class RandomIterableDataset(RandomDataset, IterableDataset):
    """Iterable version of RandomDataset"""

    def __iter__(self):
        for item in self.data:
            yield from_numpy(item)


@pytest.fixture(scope="function", params=["HyraxRandomDataset", "HyraxRandomIterableDataset"])
def loopback_hyrax(tmp_path_factory, request):
    """This generates a loopback hyrax instance
    which is configured to use the loopback model
    and a simple dataset yielding random numbers
    """
    results_dir = tmp_path_factory.mktemp(f"loopback_hyrax_{request.param}")

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


@pytest.fixture(scope="function")
def multimodal_config():
    """Create a hyrax instance with a default config setting, then update the
    config to represent a request for multimodal data."""

    return {
        "random_0": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "./in_memory_0",
            "fields": ["object_id", "image", "label"],
            "dataset_config": {
                "shape": [2, 16, 16],
            },
            "primary_id_field": "object_id",
        },
        "random_1": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "./in_memory_1",
            "fields": ["image"],
            "dataset_config": {
                "shape": [5, 16, 16],
                "seed": 4200,
            },
        },
    }


@pytest.fixture(scope="function")
def data_provider(multimodal_config):
    """Use the multimodal_config fixture to create a DataProvider instance."""
    h = hyrax.Hyrax()
    h.config["model_inputs"] = multimodal_config
    dp = DataProvider(h.config)
    return dp
