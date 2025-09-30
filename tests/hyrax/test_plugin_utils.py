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
