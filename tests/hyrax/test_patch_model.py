import numpy as np
import torch.nn as nn

from hyrax.models.model_registry import hyrax_model


@hyrax_model
class DummyModelOne(nn.Module):
    """A dummy model used to test patching of static methods like prepare_inputs"""

    def __init__(self, config, data_sample=None):
        super().__init__()
        # The optimizer needs at least one weight, so we add a dummy module here
        self.unused_module = nn.Linear(1, 1)
        self.config = config

    @staticmethod
    def prepare_inputs(x):
        """Default prepare_inputs method which just returns the input"""
        return x


@hyrax_model
class DummyModelTwo(nn.Module):
    """A dummy model used to test patching, that uses the default prepare_inputs method
    by default."""

    def __init__(self, config, data_sample=None):
        super().__init__()
        # The optimizer needs at least one weight, so we add a dummy module here
        self.unused_module = nn.Linear(1, 1)
        self.config = config


@staticmethod
def prepare_inputs(x):
    """A simple prepare_inputs method that will patch the default one on DummyModel"""
    return x * 2


def test_patch_prepare_inputs(tmp_path):
    """Test to ensure we can save and restore the prepare_inputs static method on a
    model instance correctly."""

    # Minimal config dict to define crit and optimizer for the dummy model.
    config = {
        "criterion": {"name": "torch.nn.MSELoss"},
        "optimizer": {"name": "torch.optim.SGD"},
        "torch.optim.SGD": {"lr": 0.01},
    }

    # create an instance of the dummy model
    model = DummyModelOne(config=config, data_sample=None)

    # manually update the prepare_inputs static method to be something simple
    # don't wrap this with staticmethod(...) because that would be a double wrapping.
    model.prepare_inputs = prepare_inputs

    # call model.save() to persist the model weights and prepare_inputs function.
    model.save(tmp_path / "model_weights.pth")

    # verify that the prepare_inputs file was written
    assert (tmp_path / "prepare_inputs.py").exists()

    # create a new instance of the dummy model and call .load() with the correct path
    new_model = DummyModelOne(config=config, data_sample=None)

    # verify that the new model's prepare_inputs method is the default one
    input_data = 3.0
    output_data = new_model.prepare_inputs(input_data)
    assert output_data == input_data

    # now load the saved weights and prepare_inputs method into the new model
    new_model.load(tmp_path / "model_weights.pth")

    # verify that the prepare_inputs method was restored correctly by passing some data to it.
    output_data = new_model.prepare_inputs(input_data)
    assert output_data == prepare_inputs(input_data)


def test_patch_prepare_inputs_over_default(tmp_path):
    """Test to ensure we can save and restore the prepare_inputs static method on a
    model instance where the model class makes use of the default prepare_inputs method."""

    # Minimal config dict to define crit and optimizer for the dummy model.
    config = {
        "criterion": {"name": "torch.nn.MSELoss"},
        "optimizer": {"name": "torch.optim.SGD"},
        "torch.optim.SGD": {"lr": 0.01},
    }

    # create an instance of the dummy model
    model = DummyModelTwo(config=config, data_sample=None)

    # manually update the prepare_inputs static method to be something simple
    # don't wrap this with staticmethod(...) because that would be a double wrapping.
    model.prepare_inputs = prepare_inputs

    # call model.save() to persist the model weights and prepare_inputs function.
    model.save(tmp_path / "model_weights.pth")

    # verify that the prepare_inputs file was written
    assert (tmp_path / "prepare_inputs.py").exists()

    # create a new instance of the dummy model and call .load() with the correct path
    new_model = DummyModelTwo(config=config, data_sample=None)

    # verify that the new model's prepare_inputs method is the default one
    input_data = {"data": {"image": 3}}
    output_data = new_model.prepare_inputs(input_data)
    assert output_data[0] == 3
    assert isinstance(output_data[1], np.ndarray)

    # now load the saved weights and prepare_inputs method into the new model
    new_model.load(tmp_path / "model_weights.pth")

    # verify that the prepare_inputs method was restored correctly by passing some data to it.
    input_data = 3
    output_data = new_model.prepare_inputs(input_data)
    assert output_data == prepare_inputs(input_data)
