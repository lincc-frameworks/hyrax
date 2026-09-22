import logging
import sys

import numpy as np
import pytest

import hyrax
from hyrax.context import clear_context
from hyrax.datasets.data_provider import DataProvider

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def clean_context():
    """Keep the run context from leaking between tests.

    Verbs release their own context, but tests that populate one directly - the
    vector database tests point one at a tmp_path - would otherwise leave it set
    for whatever runs next in the same worker.
    """
    clear_context()
    yield
    clear_context()


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


@pytest.fixture(scope="function")
def num_workers(request):
    """The number of DataLoader worker processes for loopback fixtures. Defaults to 0
    (in-process loading); override per-test with
    ``@pytest.mark.parametrize("num_workers", [2], indirect=True)`` to exercise
    multi-process loading.
    """
    return getattr(request, "param", 0)


@pytest.fixture(scope="function")
def loopback_hyrax(tmp_path_factory, num_workers):
    """This generates a loopback hyrax instance
    which is configured to use the loopback model
    and a simple dataset yielding random numbers
    """
    results_dir = tmp_path_factory.mktemp("loopback_hyrax_HyraxRandomDataset")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 5
    h.config["data_loader"]["num_workers"] = num_workers
    h.config["general"]["results_dir"] = str(results_dir)

    h.config["general"]["dev_mode"] = True
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "validate": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "test": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data_infer")),
                "primary_id_field": "object_id",
            },
        },
    }
    h.config["split"] = {"train": 0.6, "validate": 0.2, "test": 0.2}
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

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
        "train": {
            "random_0": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "./in_memory_0",
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
                "data_location": "./in_memory_1",
                "fields": ["image"],
                "dataset_config": {
                    "HyraxRandomDataset": {
                        "shape": [5, 16, 16],
                        "seed": 4200,
                    },
                },
            },
        },
        "infer": {
            "random_0": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "./in_memory_0",
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
                "data_location": "./in_memory_1",
                "fields": ["image"],
                "dataset_config": {
                    "HyraxRandomDataset": {
                        "shape": [5, 16, 16],
                        "seed": 4200,
                    },
                },
            },
        },
    }


@pytest.fixture(scope="function")
def data_provider(multimodal_config):
    """Use the multimodal_config fixture to create a DataProvider instance."""
    h = hyrax.Hyrax()
    h.config["data_request"] = multimodal_config
    dp = DataProvider(h.config, multimodal_config["train"])
    return dp


@pytest.fixture(scope="function")
def custom_collate_data_provider(multimodal_config):
    """Use the multimodal_config fixture to create a DataProvider instance
    with custom collate functions for each dataset."""

    from hyrax.datasets.random.hyrax_random_dataset import HyraxRandomDataset

    @staticmethod
    def collate(batch):
        """Contrived custom collate function that will return collated image
        data as well as a boolean 'mask' of the same shape.
        """
        returned_data = {}
        if "image" in batch[0]:
            batch_array = np.stack([item["image"] for item in batch], axis=0)
            returned_data["image"] = batch_array
            returned_data["image_mask"] = np.ones_like(batch_array, dtype=bool)

        if "object_id" in batch[0]:
            returned_data["object_id"] = np.stack([item["object_id"] for item in batch], axis=0)

        if "label" in batch[0]:
            returned_data["label"] = np.stack([item["label"] for item in batch], axis=0)

        return returned_data

    HyraxRandomDataset.collate = collate

    h = hyrax.Hyrax()
    h.config["data_request"] = multimodal_config
    dp = DataProvider(h.config, multimodal_config["train"])

    yield dp
    delattr(HyraxRandomDataset, "collate")


@pytest.fixture(scope="function")
def custom_field_collate_data_provider(multimodal_config):
    """Use the multimodal_config fixture to create a DataProvider instance
    with a custom collate function for the image field only."""

    from hyrax.datasets.random.hyrax_random_dataset import HyraxRandomDataset

    @staticmethod
    def collate_image(batch):
        """Contrived custom collate function that will return collated image
        data as well as a boolean 'mask' of the same shape.
        """
        returned_data = {}
        if "image" in batch[0]:
            batch_array = np.stack([item["image"] for item in batch], axis=0)
            returned_data["image"] = batch_array
            returned_data["image_mask"] = np.ones_like(batch_array, dtype=bool)

        return returned_data

    HyraxRandomDataset.collate_image = collate_image

    h = hyrax.Hyrax()
    h.config["data_request"] = multimodal_config
    dp = DataProvider(h.config, multimodal_config["train"])

    yield dp
    delattr(HyraxRandomDataset, "collate_image")


@pytest.fixture(scope="function")
def context_writing_loopback(loopback_hyrax):
    """Use the loopback_hyrax fixture as a starting point and then create a
    context-writing loopback model on top of that."""
    import json

    import torch.nn as nn

    from hyrax.context import get_context
    from hyrax.models.model_registry import hyrax_model

    @hyrax_model
    class ContextWritingLoopback(nn.Module):
        """A loopback model that persists something of its own to the results dir.

        Defined at module scope so it survives the pickling that distributed training
        and the model registry perform.
        """

        def __init__(self, config, data_sample=None):
            from functools import partial

            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)
            self.batches_seen = 0

            def load(self, weight_file):
                """This model has no meaningful weights, so loading is a noop."""
                pass

            # Overridden this way rather than as a method because Torch's __init__
            # cleverness stomps a load defined in the usual fashion. Same trick as
            # HyraxLoopback.
            self.load = partial(load, self)

        def forward(self, x):
            """Return the input unchanged."""
            if isinstance(x, (tuple, list)):
                x, _ = x
            return x

        def train_batch(self, batch):
            """Training is a noop; just count the batch."""
            self.forward(batch)
            self.batches_seen += 1
            return {"loss": 0.0}

        def validate_batch(self, batch):
            """Validation is a noop."""
            self.forward(batch)
            return {"loss": 0.0}

        def infer_batch(self, batch):
            """Inference is just a forward pass."""
            return self.forward(batch)

        def test_batch(self, batch):
            """Testing is a noop; just count the batch."""
            self.forward(batch)
            self.batches_seen += 1
            return {"loss": 0.0}

        def train_post_epoch(self):
            """Persist something that does not fit the TensorBoard/MLflow paradigm."""
            self._write_notes()

        def test_post_epoch(self):
            """Same, at the end of the test pass."""
            self._write_notes()

        def _write_notes(self):
            """Write accumulated state into the results dir of the run underway.

            Read the context here rather than stashing a handle in __init__: this is
            the form that reports the right rank under distributed training, where a
            handle taken in the parent is pickled into each child as a stale copy.
            """
            context = get_context()
            outfile = context["results_dir"] / "my_notes.json"
            with open(outfile, "w") as f:
                json.dump({"batches_seen": self.batches_seen, "verb": context["verb"]}, f)

        @staticmethod
        def prepare_inputs(data_dict):
            """Simple input prep, matching HyraxLoopback."""
            import numpy as np

            data = data_dict.get("data")
            image = data.get("image", np.array([], dtype=np.float32))
            label = data.get("label", np.array([], dtype=np.float32))
            return (image, label)

    h, _ = loopback_hyrax
    h.config["model"]["name"] = "ContextWritingLoopback"

    return h
