import torch.nn as nn

from hyrax import Hyrax
from hyrax.models.model_registry import hyrax_model


@hyrax_model
class DummyModel(nn.Module):
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


def to_tensor(x):
    """A simple to_tensor method that will patch the default one on DummyModel"""
    return x * 2


def test_patch_to_tensor(tmp_path):
    """Test to ensure we can save and restore the to_tensor static method on a
    model instance correctly."""

    # Used to get a config dict to define crit and optimizer for the dummy model.
    h = Hyrax()

    # create an instance of the dummy model
    model = DummyModel(config=h.config, data_sample=None)

    # manually update the to_tensor static method to be something simple
    model.to_tensor = staticmethod(to_tensor)

    # call model.save() to put the weights and to_tensor method into a state dict
    model.save(tmp_path / "model_weights.pth")

    # create a new instance of the HyraxLoopback model and call .load() with the correct path
    new_model = DummyModel(config=h.config, data_sample=None)

    # verify that the new model's to_tensor method is the default one
    input_data = 3.0
    output_data = new_model.to_tensor(input_data)
    assert output_data == input_data

    # now load the saved weights and to_tensor method into the new model
    new_model.load(tmp_path / "model_weights.pth")

    # verify that the to_tensor method was restored correctly by passing some data to it.
    output_data = new_model.to_tensor(input_data)
    assert output_data == to_tensor(input_data)
