"""Tests for prepare_inputs migration and backward compatibility."""

import logging

import numpy as np
import torch.nn as nn

from hyrax import Hyrax
from hyrax.models.model_registry import hyrax_model


def test_model_with_only_to_tensor():
    """Test that a model with only to_tensor gets prepare_inputs created from it."""

    @hyrax_model
    class TestModelWithToTensor(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)

        @staticmethod
        def to_tensor(data_dict):
            """Old to_tensor method."""
            return data_dict.get("data", {}).get("image", np.array([]))

        def forward(self, x):
            return x

        def train_step(self, batch):
            return {"loss": 0.0}

    h = Hyrax()
    h.set_config("model.name", "TestModelWithToTensor")
    h.set_config("optimizer.name", "torch.optim.SGD")
    h.set_config("criterion.name", "torch.nn.MSELoss")

    model = TestModelWithToTensor(h.config)

    # Model should have prepare_inputs created from to_tensor
    assert hasattr(model, "prepare_inputs")
    assert hasattr(model, "to_tensor")

    # Test that prepare_inputs works
    test_data = {"data": {"image": np.array([1, 2, 3])}}
    result = model.prepare_inputs(test_data)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_model_without_prepare_inputs_or_to_tensor():
    """Test that a model without prepare_inputs or to_tensor gets default prepare_inputs."""

    @hyrax_model
    class TestModelNoInputMethod(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)

        def forward(self, x):
            return x

        def train_step(self, batch):
            return {"loss": 0.0}

    h = Hyrax()
    h.set_config("model.name", "TestModelNoInputMethod")
    h.set_config("optimizer.name", "torch.optim.SGD")
    h.set_config("criterion.name", "torch.nn.MSELoss")

    model = TestModelNoInputMethod(h.config)

    # Model should have default prepare_inputs
    assert hasattr(model, "prepare_inputs")

    # Test that default prepare_inputs works with proper data structure
    test_data = {"data": {"image": np.array([1, 2, 3]), "label": np.array([0])}}
    result = model.prepare_inputs(test_data)
    assert isinstance(result, tuple)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(result[1], np.array([0]))


def test_model_without_prepare_inputs_or_to_tensor_raises_error():
    """Test that default prepare_inputs raises error when data key is missing."""

    @hyrax_model
    class TestModelNoInputMethod(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)

        def forward(self, x):
            return x

        def train_step(self, batch):
            return {"loss": 0.0}

    h = Hyrax()
    h.set_config("model.name", "TestModelNoInputMethod")
    h.set_config("optimizer.name", "torch.optim.SGD")
    h.set_config("criterion.name", "torch.nn.MSELoss")

    model = TestModelNoInputMethod(h.config)

    # Test that default prepare_inputs raises error with improper data structure
    test_data = {"wrong_key": {"image": np.array([1, 2, 3])}}
    try:
        model.prepare_inputs(test_data)
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "Hyrax couldn't find a 'data' key" in str(e)
        assert "prepare_inputs" in str(e)


def test_load_model_with_to_tensor_file(tmp_path, caplog):
    """Test loading a model that has to_tensor.py file but no prepare_inputs.py."""

    @hyrax_model
    class TestModelForLoading(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)

        @staticmethod
        def prepare_inputs(data_dict):
            return data_dict.get("image", np.array([]))

        def forward(self, x):
            return x

        def train_step(self, batch):
            return {"loss": 0.0}

    h = Hyrax()
    h.set_config("model.name", "TestModelForLoading")
    h.set_config("optimizer.name", "torch.optim.SGD")
    h.set_config("criterion.name", "torch.nn.MSELoss")

    # Create a model instance
    model = TestModelForLoading(h.config)

    # Save the model
    save_path = tmp_path / "model.pth"
    model.save(save_path)

    # Manually rename prepare_inputs.py to to_tensor.py to simulate old model.
    # It needs to rename more than the file.  The method name inside must
    # also be renamed.
    prepare_inputs_file = tmp_path / "prepare_inputs.py"
    to_tensor_file = tmp_path / "to_tensor.py"
    if prepare_inputs_file.exists():
        prepare_inputs_file.rename(to_tensor_file)
        src = to_tensor_file.read_text()
        src = src.replace("def prepare_inputs", "def to_tensor")
        to_tensor_file.write_text(src)

    # Create new model instance and load
    new_model = TestModelForLoading(h.config)

    with caplog.at_level(logging.WARNING):
        new_model.load(save_path)
        # Should warn about finding to_tensor
        assert "Found to_tensor function" in caplog.text
        assert "deprecated" in caplog.text

    # Should have prepare_inputs loaded from to_tensor
    assert hasattr(new_model, "prepare_inputs")


def test_load_model_without_prepare_inputs_or_to_tensor_file(tmp_path, caplog):
    """Test loading a model when neither prepare_inputs.py nor to_tensor.py exists."""

    @hyrax_model
    class TestModelForLoading(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)

        @staticmethod
        def prepare_inputs(data_dict):
            return data_dict.get("image", np.array([]))

        def forward(self, x):
            return x

        def train_step(self, batch):
            return {"loss": 0.0}

    h = Hyrax()
    h.set_config("model.name", "TestModelForLoading")
    h.set_config("optimizer.name", "torch.optim.SGD")
    h.set_config("criterion.name", "torch.nn.MSELoss")

    # Create a model instance
    model = TestModelForLoading(h.config)

    # Save the model
    save_path = tmp_path / "model.pth"
    model.save(save_path)

    # Remove both prepare_inputs.py and to_tensor.py files
    prepare_inputs_file = tmp_path / "prepare_inputs.py"
    to_tensor_file = tmp_path / "to_tensor.py"
    if prepare_inputs_file.exists():
        prepare_inputs_file.unlink()
    if to_tensor_file.exists():
        to_tensor_file.unlink()

    # Create new model instance and load
    new_model = TestModelForLoading(h.config)

    with caplog.at_level(logging.WARNING):
        new_model.load(save_path)
        # Should warn about not finding either file
        assert "Could not find prepare_inputs or to_tensor function" in caplog.text

    # Should still have prepare_inputs from the class definition
    assert hasattr(new_model, "prepare_inputs")
