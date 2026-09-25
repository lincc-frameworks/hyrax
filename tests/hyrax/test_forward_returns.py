import torch

import hyrax
from hyrax.models.hyrax_loopback import HyraxLoopback
from hyrax.models.model_registry import hyrax_model

RANDOM_DATASET_CONFIG = {
    "data": {
        "dataset_class": "HyraxRandomDataset",
        "data_location": "./data/test",
        "primary_id_field": "object_id",
    }
}


@hyrax_model
class DummyModelDictReturn(HyraxLoopback):
    """Dummy model that returns a dictionary from the forward pass."""

    def forward(self, x):
        """Forward pass that returns a dictionary."""
        output = super().forward(x)
        return {"output": output, "square": torch.ones((output.shape[0], 2, 2))}

    @staticmethod
    def prepare_inputs(data_dict):
        """Prepare the inputs for the forward pass."""
        import numpy as np

        data = data_dict["data"]
        image = np.asarray(data["image"], dtype=np.float32)
        label = np.asarray(data.get("label", []), dtype=np.int64)

        return (image, label)

    def infer_batch(self, batch):
        """Inference is just a forward pass."""
        return self.forward(batch)


# (the current operational implementation)
def test_return_tensor():
    """Test that a model that returns a tensor can be used in the forward pass."""
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["data_request"] = {
        "train": RANDOM_DATASET_CONFIG,
        "infer": RANDOM_DATASET_CONFIG,
    }

    h.config["data_set"]["HyraxRandomDataset"]["size"] = 100
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 24601
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [10]
    h.config["data_set"]["HyraxRandomDataset"]["provided_labels"] = [0]

    h.config["train"]["epochs"] = 1

    h.train()
    infda = h.infer()

    assert len(infda) == 100
    assert infda[0]["data"].shape == (10,)
    assert hasattr(infda, "get_data")


def test_return_dict():
    """Test that a model that returns a dictionary can be used in the forward pass."""
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "DummyModelDictReturn"
    h.config["data_request"] = {
        "train": RANDOM_DATASET_CONFIG,
        "infer": RANDOM_DATASET_CONFIG,
    }

    h.config["data_set"]["HyraxRandomDataset"]["size"] = 100
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 24601
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [10]
    h.config["data_set"]["HyraxRandomDataset"]["provided_labels"] = [0]

    h.config["train"]["epochs"] = 1

    h.train()
    infda = h.infer()

    assert infda
    assert infda[0]["output"].shape == (10,)
    assert hasattr(infda, "get_output")

    # makes sure we maintain shape
    assert infda[0]["square"].shape == (2, 2)
    assert hasattr(infda, "get_square")
