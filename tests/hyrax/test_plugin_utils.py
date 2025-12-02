import logging

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


def test_fetch_model_class_false_model():
    """Test that the fetch_model_class function raises an error when model
    is set to false in the configuration."""

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


def test_fetch_model_class_false_logs_registered_models(caplog):
    """Test that the fetch_model_class function logs registered models when
    model is set to false."""

    config = {"model": {"name": ""}}

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            fetch_model_class(config)

    # Check that the error message contains expected information
    assert "No model name was provided" in caplog.text
    assert "h.set_config('model.name'" in caplog.text
    assert "Currently registered models:" in caplog.text


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
    """Test that _torch_load properly handles device remapping via map_location.

    This test verifies that the fix works by checking that loaded tensors are on
    the expected device (as determined by idist.device()). This tests the actual
    problem (device mismatch handling) without mocking torch's interface, making
    it robust to future PyTorch API changes.

    While we can't create a true GPU->CPU scenario in CPU-only CI, this test ensures
    that whatever device is available, the loaded state dict tensors match it.
    """
    import ignite.distributed as idist
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

    # Get the expected device from idist (same as _torch_load uses)
    expected_device = idist.device()

    # Create and save a model
    model = SimpleModel(config)
    weights_path = tmp_path / "test_weights.pth"
    model.save(weights_path)

    # Create a new model instance and load the weights
    new_model = SimpleModel(config)
    new_model.load(weights_path)

    # Verify that all loaded tensors are on the expected device
    # This is the actual fix: map_location ensures tensors are on the right device
    for key, tensor in new_model.state_dict().items():
        assert tensor.device.type == expected_device.type, (
            f"Tensor {key} is on device {tensor.device}, expected {expected_device}. "
            "This indicates map_location is not working correctly."
        )

    # Verify that the weights were loaded correctly (functional test)
    for key in model.state_dict():
        assert torch.allclose(
            model.state_dict()[key].to(expected_device), new_model.state_dict()[key].to(expected_device)
        )
