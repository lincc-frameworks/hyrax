import torch.nn as nn

import hyrax
from hyrax.models.model_registry import hyrax_model

RANDOM_DATASET_CONFIG = {
    "data": {
        "dataset_class": "HyraxRandomDataset",
        "data_location": "./data/test",
        "primary_id_field": "object_id",
    }
}


@hyrax_model
class DummyModel(nn.Module):
    """Dummy model that returns a tensor from the forward pass."""

    def __init__(self, config, data_sample=None):
        super().__init__()
        self.config = config

        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        """Forward pass that returns a tensor."""
        return self.linear(x)

    @staticmethod
    def prepare_inputs(data_dict):
        """Prepare the inputs for the forward pass."""
        import numpy as np

        data = data_dict["data"]
        image = np.asarray(data["image"], dtype=np.float32)
        label = np.asarray(data.get("label", []), dtype=np.int64)

        return (image, label)

    def train_batch(self, batch):
        """Train step that returns a loss value."""
        data, labels = batch
        outputs = self(data)
        loss = self.criterion(outputs, labels)
        return {"loss": loss.item()}

    def validate_batch(self, batch):
        """Validation step that returns a loss value."""
        data, labels = batch
        outputs = self(data)
        loss = self.criterion(outputs, labels)
        return {"loss": loss.item()}

    def infer_batch(self, batch):
        """Inference step that returns the model output."""
        data, _ = batch
        return self(data)


@hyrax_model
class DummyModelDictReturn(DummyModel):
    """Dummy model that returns a dictionary from the forward pass."""

    def forward(self, x):
        """Forward pass that returns a dictionary."""
        output = super().forward(x)
        return {"output": output}

    @staticmethod
    def prepare_inputs(data_dict):
        """Prepare the inputs for the forward pass."""
        import numpy as np

        data = data_dict["data"]
        image = np.asarray(data["image"], dtype=np.float32)
        label = np.asarray(data.get("label", []), dtype=np.int64)

        return (image, label)

    def train_batch(self, batch):
        """Train step that returns a loss value."""
        data, labels = batch
        outputs = self(data)
        loss = self.criterion(outputs["output"], labels)
        return {"loss": loss.item()}

    def validate_batch(self, batch):
        """Validation step that returns a loss value."""
        data, labels = batch
        outputs = self(data)
        loss = self.criterion(outputs["output"], labels)
        return {"loss": loss.item()}


# (the current operational implementation)
def test_return_tensor():
    """Test that a model that returns a tensor can be used in the forward pass."""
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "DummyModel"
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
    assert infda[0]["data"].shape == (1,)


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
    assert infda[0]["output"].shape == (1,)
