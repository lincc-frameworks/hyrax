import pytest

from hyrax import plugin_utils
from hyrax.models import hyrax_model
from hyrax.models.model_registry import fetch_model_class


def test_import_module_from_string():
    """Test the import_module_from_string function."""
    module_path = "builtins.BaseException"

    model_cls = plugin_utils.import_module_from_string(module_path)

    assert model_cls.__name__ == "BaseException"


def test_import_module_from_string_no_base_module():
    """Test that the import_module_from_string function raises an error when
    the base module is not found."""

    module_path = "nonexistent.BaseException"

    with pytest.raises(ModuleNotFoundError) as excinfo:
        plugin_utils.import_module_from_string(module_path)

    assert "Module nonexistent not found" in str(excinfo.value)


def test_import_module_from_string_no_submodule():
    """Test that the import_module_from_string function raises an error when
    a submodule is not found."""

    module_path = "builtins.nonexistent.BaseException"

    with pytest.raises(ModuleNotFoundError) as excinfo:
        plugin_utils.import_module_from_string(module_path)

    assert "Module builtins.nonexistent not found" in str(excinfo.value)


def test_import_module_from_string_no_class():
    """Test that the import_module_from_string function raises an error when
    a class is not found."""

    module_path = "builtins.Nonexistent"

    with pytest.raises(AttributeError) as excinfo:
        plugin_utils.import_module_from_string(module_path)

    assert "Unable to find Nonexistent in module" in str(excinfo.value)


def test_fetch_model_class():
    """Test the fetch_model_class function."""
    config = {"model": {"name": "builtins.BaseException"}}

    model_cls = fetch_model_class(config)

    assert model_cls.__name__ == "BaseException"


def test_fetch_model_class_no_model():
    """Test that the fetch_model_class function raises an error when no model
    is specified in the configuration."""

    config = {"model": {"name": ""}}

    with pytest.raises(RuntimeError) as excinfo:
        fetch_model_class(config)

    assert "A model class name or path must be provided" in str(excinfo.value)


def test_fetch_model_class_no_model_cls():
    """Test that an exception is raised when a non-existent model class is requested."""

    config = {"model": {"name": "builtins.Nonexistent"}}

    with pytest.raises(AttributeError) as excinfo:
        fetch_model_class(config)

    assert "Unable to find Nonexistent in module" in str(excinfo.value)


def test_fetch_model_class_not_in_registry():
    """Test that an exception is raised when a model is requested that is not in the registry."""

    config = {"model": {"name": "Nonexistent"}}

    with pytest.raises(ValueError) as excinfo:
        fetch_model_class(config)

    assert "not found in registry and is not a full import path" in str(excinfo.value)


def test_fetch_model_class_in_registry():
    """Test that a model class is returned when it is in the registry."""

    # make a no-op model that will be added to the model registry
    @hyrax_model
    class NewClass:
        pass

    config = {"model": {"name": "NewClass"}}
    model_cls = fetch_model_class(config)

    assert model_cls.__name__ == "NewClass"


def test_torch_load_with_map_location(tmp_path):
    """Test that _torch_load uses map_location parameter to handle device remapping.

    This test verifies that the fix for GPU->CPU loading works by:
    1. Mocking torch.load to verify map_location is passed
    2. Testing that the model loads successfully with the map_location parameter

    The actual GPU->CPU scenario is difficult to test in CI (CPU-only can't create GPU
    state dicts, GPU CI won't experience the error), so we verify that the fix
    (adding map_location parameter) is correctly implemented.
    """
    from unittest.mock import patch

    import torch
    import torch.nn as nn

    from hyrax.models.model_registry import hyrax_model

    # Create a simple model
    @hyrax_model
    class SimpleModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, batch):
            return {"loss": 0.0}

    # Create config
    config = {
        "criterion": {"name": "torch.nn.MSELoss"},
        "optimizer": {"name": "torch.optim.SGD"},
        "torch.optim.SGD": {"lr": 0.01},
    }

    # Create model instance and save
    model = SimpleModel(config)
    weights_path = tmp_path / "test_weights.pth"
    model.save(weights_path)

    # Create a new model instance
    new_model = SimpleModel(config)

    # Mock torch.load to verify map_location is used
    original_torch_load = torch.load

    def mock_torch_load(path, *args, **kwargs):
        # Verify that map_location parameter is present
        assert "map_location" in kwargs, "map_location parameter must be passed to torch.load"
        # Call the original function
        return original_torch_load(path, *args, **kwargs)

    # Patch torch.load and load the model
    with patch("torch.load", side_effect=mock_torch_load) as mock_load:
        new_model.load(weights_path)

        # Verify torch.load was called
        assert mock_load.called, "torch.load should have been called"

        # Verify map_location was in the call
        call_kwargs = mock_load.call_args[1]
        assert "map_location" in call_kwargs, "map_location must be passed to torch.load"

    # Verify that the weights were loaded correctly
    for key in model.state_dict():
        assert torch.allclose(model.state_dict()[key], new_model.state_dict()[key])
