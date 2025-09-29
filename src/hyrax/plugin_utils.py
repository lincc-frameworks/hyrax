import importlib
from importlib import util as importlib_util
from typing import Any, Optional, TypeVar, Union

T = TypeVar("T")


def get_or_load_class(class_name: str, registry: Optional[dict[str, T]] = None) -> Union[T, Any]:
    """Given a configuration dictionary and a registry dictionary, attempt to return
    the requested class either from the registry or by dynamically importing it.

    Parameters
    ----------
    class_name : str
        The name of the class to load.
    registry : dict
        The registry dictionary of <class name> : <class type> pairs.

    Returns
    -------
    type
        The returned class to be instantiated

    """

    if registry and class_name in registry:
        returned_class = registry[class_name]
    else:
        #! I'm not a big fan of this check here. It would be nice if this was
        #! handled more gracefully elsewhere.
        if "." not in class_name:
            raise ValueError(f"Class name {class_name} not found in registry and is not a full import path.")
        returned_class = import_module_from_string(class_name)
        # if the class was loaded successfully, add it to the provided registry
        if registry:
            update_registry(registry, class_name, returned_class)

    return returned_class


def import_module_from_string(module_path: str) -> Any:
    """Dynamically import a module from a string.

    Parameters
    ----------
    module_path : str
        The import spec for the requested class. Should be of the form:
        "module.submodule.ClassName"

    Returns
    -------
    returned_cls : type
        The class type that was loaded.

    Raises
    ------
    AttributeError
        If the class is not found in the module that is loaded.
    ModuleNotFoundError
        If the module is not found using the provided import spec.
    """

    if "." not in module_path:
        raise ValueError(f"Invalid module path: {module_path}. Expected format: 'module.submodule.ClassName'")

    module_name, class_name = module_path.rsplit(".", 1)
    returned_cls = None

    try:
        # Attempt to find the module spec, i.e. `module.submodule.`.
        # Will raise exception if `submodule`, 'subsubmodule', etc. is not found.
        importlib_util.find_spec(module_name)

        # `importlib_util.find_spec()` will return None if `module` is not found.
        if (importlib_util.find_spec(module_name)) is not None:
            # Load the requested module
            module = importlib.import_module(module_name)

            # Check if the requested class is in the module
            if hasattr(module, class_name):
                returned_cls = getattr(module, class_name)
            else:
                raise AttributeError(f"Unable to find {class_name} in module {module_name}")

        # Raise an exception if the base module of the spec is not found
        else:
            raise ModuleNotFoundError(f"Module {module_name} not found")

    # Exception raised when a submodule of the spec is not found
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Module {module_name} not found") from exc

    return returned_cls


def update_registry(registry: dict, name: str, class_type: type):
    """Add a class to a given registry dictionary.

    Parameters
    ----------
    registry : dict
        The registry to update.
    name : str
        The name of the class.
    class_type : type
        The class type to be instantiated.
    """

    registry.update({name: class_type})
