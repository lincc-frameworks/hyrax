import pytest
from fibad import plugin_utils
from fibad.models import fibad_model


def test_import_module_from_string():
    """Test the import_module_from_string function."""
    module_path = "builtins.BaseException"

    model_cls = plugin_utils._import_module_from_string(module_path)

    assert model_cls.__name__ == "BaseException"


def test_import_module_from_string_no_base_module():
    """Test that the import_module_from_string function raises an error when
    the base module is not found."""

    module_path = "nonexistent.BaseException"

    with pytest.raises(ModuleNotFoundError) as excinfo:
        plugin_utils._import_module_from_string(module_path)

    assert "Module nonexistent not found" in str(excinfo.value)


def test_import_module_from_string_no_submodule():
    """Test that the import_module_from_string function raises an error when
    a submodule is not found."""

    module_path = "builtins.nonexistent.BaseException"

    with pytest.raises(ModuleNotFoundError) as excinfo:
        plugin_utils._import_module_from_string(module_path)

    assert "Module builtins.nonexistent not found" in str(excinfo.value)


def test_import_module_from_string_no_class():
    """Test that the import_module_from_string function raises an error when
    a class is not found."""

    module_path = "builtins.Nonexistent"

    with pytest.raises(AttributeError) as excinfo:
        plugin_utils._import_module_from_string(module_path)

    assert "Model class Nonexistent not found" in str(excinfo.value)


def test_fetch_model_class():
    """Test the fetch_model_class function."""
    config = {"train": {"model_cls": "builtins.BaseException"}}

    model_cls = plugin_utils.fetch_model_class(config)

    assert model_cls.__name__ == "BaseException"


def test_fetch_model_class_no_model():
    """Test that the fetch_model_class function raises an error when no model
    is specified in the configuration."""

    config = {"train": {}}

    with pytest.raises(ValueError) as excinfo:
        plugin_utils.fetch_model_class(config)

    assert "No model specified in the runtime configuration" in str(excinfo.value)


def test_fetch_model_class_no_model_cls():
    """Test that an exception is raised when a non-existent model class is requested."""

    config = {"train": {"model_cls": "builtins.Nonexistent"}}

    with pytest.raises(AttributeError) as excinfo:
        plugin_utils.fetch_model_class(config)

    assert "Model class Nonexistent not found" in str(excinfo.value)


def test_fetch_model_class_not_in_registry():
    """Test that an exception is raised when a model is requested that is not in the registry."""

    config = {"train": {"model_name": "Nonexistent"}}

    with pytest.raises(ValueError) as excinfo:
        plugin_utils.fetch_model_class(config)

    assert "Model not found in model registry: Nonexistent" in str(excinfo.value)


def test_fetch_model_class_in_registry():
    """Test that a model class is returned when it is in the registry."""

    # make a no-op model that will be added to the model registry
    @fibad_model
    class NewClass:
        pass

    config = {"train": {"model_name": "NewClass"}}
    model_cls = plugin_utils.fetch_model_class(config)

    assert model_cls.__name__ == "NewClass"
