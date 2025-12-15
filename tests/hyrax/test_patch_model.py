import numpy as np
import torch.nn as nn

from hyrax.models.model_registry import hyrax_model


@hyrax_model
class DummyModelOne(nn.Module):
    """A dummy model used to test patching of static methods like to_tensor"""

    def __init__(self, config, data_sample=None):
        super().__init__()
        # The optimizer needs at least one weight, so we add a dummy module here
        self.unused_module = nn.Linear(1, 1)
        self.config = config

    @staticmethod
    def to_tensor(x):
        """Default to_tensor method which just returns the input"""
        return x


@hyrax_model
class DummyModelTwo(nn.Module):
    """A dummy model used to test patching, that uses the default to_tensor method
    by default."""

    def __init__(self, config, data_sample=None):
        super().__init__()
        # The optimizer needs at least one weight, so we add a dummy module here
        self.unused_module = nn.Linear(1, 1)
        self.config = config


@staticmethod
def to_tensor(x):
    """A simple to_tensor method that will patch the default one on DummyModel"""
    return x * 2


def test_patch_to_tensor(tmp_path):
    """Test to ensure we can save and restore the to_tensor static method on a
    model instance correctly."""

    # Minimal config dict to define crit and optimizer for the dummy model.
    config = {
        "criterion": {"name": "torch.nn.MSELoss"},
        "optimizer": {"name": "torch.optim.SGD"},
        "torch.optim.SGD": {"lr": 0.01},
    }

    # create an instance of the dummy model
    model = DummyModelOne(config=config, data_sample=None)

    # manually update the to_tensor static method to be something simple
    # don't wrap this with staticmethod(...) because that would be a double wrapping.
    model.to_tensor = to_tensor

    # call model.save() to persist the model weights and to_tensor function.
    model.save(tmp_path / "model_weights.pth")

    # verify that the to_tensor file was written
    assert (tmp_path / "to_tensor.py").exists()

    # create a new instance of the dummy model and call .load() with the correct path
    new_model = DummyModelOne(config=config, data_sample=None)

    # verify that the new model's to_tensor method is the default one
    input_data = 3.0
    output_data = new_model.to_tensor(input_data)
    assert output_data == input_data

    # now load the saved weights and to_tensor method into the new model
    new_model.load(tmp_path / "model_weights.pth")

    # verify that the to_tensor method was restored correctly by passing some data to it.
    output_data = new_model.to_tensor(input_data)
    assert output_data == to_tensor(input_data)


def test_patch_to_tensor_over_default(tmp_path):
    """Test to ensure we can save and restore the to_tensor static method on a
    model instance where the model class makes use of the default to_tensor method."""

    # Minimal config dict to define crit and optimizer for the dummy model.
    config = {
        "criterion": {"name": "torch.nn.MSELoss"},
        "optimizer": {"name": "torch.optim.SGD"},
        "torch.optim.SGD": {"lr": 0.01},
    }

    # create an instance of the dummy model
    model = DummyModelTwo(config=config, data_sample=None)

    # manually update the to_tensor static method to be something simple
    # don't wrap this with staticmethod(...) because that would be a double wrapping.
    model.to_tensor = to_tensor

    # call model.save() to persist the model weights and to_tensor function.
    model.save(tmp_path / "model_weights.pth")

    # verify that the to_tensor file was written
    assert (tmp_path / "to_tensor.py").exists()

    # create a new instance of the dummy model and call .load() with the correct path
    new_model = DummyModelTwo(config=config, data_sample=None)

    # verify that the new model's to_tensor method is the default one
    input_data = {"data": {"image": 3}}
    output_data = new_model.to_tensor(input_data)
    assert output_data[0] == 3
    assert isinstance(output_data[1], np.ndarray)

    # now load the saved weights and to_tensor method into the new model
    new_model.load(tmp_path / "model_weights.pth")

    # verify that the to_tensor method was restored correctly by passing some data to it.
    input_data = 3
    output_data = new_model.to_tensor(input_data)
    assert output_data == to_tensor(input_data)
