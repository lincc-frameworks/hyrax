import importlib
import inspect
import logging
import textwrap
from pathlib import Path
from typing import Any, cast

import torch.nn as nn
from torch import Tensor, as_tensor

from hyrax.plugin_utils import get_or_load_class, update_registry

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def _torch_save(self: nn.Module, save_path: Path):
    import torch

    # save the model weights
    torch.save(self.state_dict(), save_path)

    # Save the to_tensor static method in a .py file alongside the model weights
    with open(save_path.parent / "to_tensor.py", "w") as f:
        if self.to_tensor.__name__ != "to_tensor":
            logger.warning(
                "It is highly recommended to name your to_tensor method 'to_tensor'. "
                f"Currently it is {self.to_tensor.__name__}."
            )
        try:
            f.write(textwrap.dedent(inspect.getsource(self.to_tensor)))
        except (OSError, TypeError) as e:
            logger.warning(f"Could not retrieve source for model.to_tensor: {e}")
            f.write("# Source code for model.to_tensor could not be retrieved.\n")


def _torch_load(self: nn.Module, load_path: Path):
    import ignite.distributed as idist
    import torch

    # Use ignite's device detection which handles distributed training and device availability
    # This allows models trained on GPU to be loaded on CPU-only machines
    device = idist.device()
    state = torch.load(load_path, weights_only=True, map_location=device)

    self.load_state_dict(state, assign=True)

    # Monkey patch the to_tensor static method from the saved .py file
    # We prefer to use the .py file, because that allows using Python's import system
    # and allows us to inspect and save the function again as needed.
    to_tensor_source_path = load_path.parent / "to_tensor.py"
    if to_tensor_source_path.exists():
        spec = importlib.util.spec_from_file_location("to_tensor", to_tensor_source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore

        to_tensor = None
        # We expect people to name their functions "to_tensor"
        if hasattr(module, "to_tensor"):
            to_tensor = module.to_tensor
        # If using the Hyrax default, it will be called "default_to_tensor"
        elif hasattr(module, "default_to_tensor"):
            to_tensor = module.default_to_tensor

        # If neither to_tensor or default_to_tensor was found, log a warning,
        # and use the existing model's to_tensor method.
        if not to_tensor:
            logger.warning(
                f"Could not find to_tensor function in {to_tensor_source_path}. "
                "Using the model's existing to_tensor method."
            )
        else:
            if isinstance(to_tensor, staticmethod):
                self.to_tensor = to_tensor
            else:
                self.to_tensor = staticmethod(to_tensor)


def _torch_criterion(self: nn.Module):
    """Load the criterion class using the name defined in the config and
    instantiate it with the arguments defined in the config."""

    config = cast(dict[str, Any], self.config)

    # Load the class and get any parameters from the config dictionary
    criterion_name = config["criterion"]["name"]
    if not criterion_name:
        logger.warning("No criterion specified in config or self.criterion in model.")
        return None
    criterion_cls = get_or_load_class(criterion_name)

    arguments = {}
    if criterion_name in config:
        arguments = config[criterion_name]

    # Print some debugging info about the criterion function and parameters used
    log_string = f"Setting model's self.criterion from config: {criterion_name} "
    if arguments:
        log_string += f"with arguments: {arguments}."
    else:
        log_string += "with default arguments."
    logger.info(log_string)

    return criterion_cls(**arguments)


def _torch_optimizer(self: nn.Module):
    """Load the optimizer class using the name defined in the config and
    instantiate it with the arguments defined in the config."""

    config = cast(dict[str, Any], self.config)

    # Load the class and get any parameters from the config dictionary
    optimizer_name = config["optimizer"]["name"]
    if not optimizer_name:
        logger.warning("No optimizer specified in config or self.optimizer in model.")
        return None

    optimizer_cls = get_or_load_class(optimizer_name)

    arguments = {}
    if optimizer_name in config:
        arguments = config[optimizer_name]

    # Print some debugging info about the optimizer function and parameters used
    log_string = f"Setting model's self.optimizer from config: {optimizer_name} "
    if arguments:
        log_string += f"with arguments: {arguments}."
    else:
        log_string += "with default arguments."
    logger.info(log_string)

    return optimizer_cls(self.parameters(), **arguments)


def hyrax_model(cls):
    """Decorator to register a model with the model registry, and to add common interface functions

    Returns
    -------
    type
        The class with additional interface functions.
    """

    if issubclass(cls, nn.Module):
        cls.save = _torch_save
        cls.load = _torch_load

    original_init = cls.__init__

    def wrapped_init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)

        if not hasattr(self, "optimizer"):
            self.optimizer = _torch_optimizer(self)
        else:
            if config["optimizer"]["name"]:
                logger.warning(
                    "Both model and config define an optimizer. "
                    "Hyrax will use self.optimizer defined in the model."
                )
            opt_name = f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}"
            logger.info(f"Using self.optimizer defined in model: {opt_name}")

        if not hasattr(self, "criterion"):
            self.criterion = _torch_criterion(self)
        else:
            if config["criterion"]["name"]:
                logger.warning(
                    "Both model and config define a criterion. "
                    "Hyrax will use self.criterion defined in the model."
                )
            crit_name = f"{type(self.criterion).__module__}.{type(self.criterion).__qualname__}"
            logger.info(f"Using self.criterion defined in model: {crit_name}")

    cls.__init__ = wrapped_init

    def default_to_tensor(data_dict):
        data = data_dict.get("data")
        if data is None:
            msg = "Hyrax couldn't find a 'data' key in the data dictionaries from your dataset.\n"
            msg += f"We recommend you implement a function on {cls.__name__} to unpack the appropriate\n"
            msg += "value(s) from the dictionary your dataset is returning:\n\n"
            msg += f"class {cls.__name__}:\n\n"
            msg += "    @staticmethod\n"
            msg += "    def to_tensor(data_dict) -> Tensor:\n"
            msg += "        <Your implementation goes here>\n\n"
            raise RuntimeError(msg)

        if "image" in data and not isinstance(data["image"], Tensor):
            data["image"] = as_tensor(data["image"])
        if isinstance(data.get("image"), Tensor):
            if "label" in data:
                return (data["image"], data["label"])
            else:
                return data["image"]
        else:
            msg = "Hyrax couldn't find an image in the data dictionaries from your dataset.\n"
            msg += f"We recommend you implement a function on {cls.__name__} to unpack the appropriate\n"
            msg += "value(s) from the dictionary your dataset is returning:\n\n"
            msg += f"class {cls.__name__}:\n\n"
            msg += "    @staticmethod\n"
            msg += "    def to_tensor(data_dict) -> Tensor:\n"
            msg += "        <Your implementation goes here>\n\n"
            raise RuntimeError(msg)

    if not hasattr(cls, "to_tensor"):
        cls.to_tensor = staticmethod(default_to_tensor)

    if not isinstance(vars(cls)["to_tensor"], staticmethod):
        msg = f"You must implement to_tensor() in {cls.__name__} as\n\n"
        msg += "@staticmethod\n"
        msg += "to_tensor(data_dict: dict) -> torch.Tensor:\n"
        msg += "    <Your implementation goes here>\n"
        raise RuntimeError(msg)

    required_methods = ["train_step", "forward", "__init__", "to_tensor"]
    for name in required_methods:
        if not hasattr(cls, name):
            logger.error(f"Hyrax model {cls.__name__} missing required method {name}.")

    update_registry(MODEL_REGISTRY, cls.__name__, cls)
    return cls


def fetch_model_class(runtime_config: dict) -> type[nn.Module]:
    """Fetch the model class from the model registry.

    Parameters
    ----------
    runtime_config : dict
        The runtime configuration dictionary.

    Returns
    -------
    type
        The model class.

    Raises
    ------
    ValueError
        If a built in model was requested, but not found in the model registry.
    ValueError
        If no model was specified in the runtime configuration.
    """

    model_name = runtime_config["model"]["name"] if runtime_config["model"]["name"] else None
    model_cls = None

    if not model_name:
        model_list = "\n".join([f"  - {model}" for model in sorted(MODEL_REGISTRY.keys())])
        logger.error(
            "No model name was provided in the configuration. "
            "You must specify a model to use before running Hyrax.\n\n"
            "To set a model, use: h.set_config('model.name', '<model_name>')\n"
            "<model_name> can be one of the following registered models or a path to a custom model class "
            "e.g. 'HyraxCNN' or 'my_package.my_module.MyModelClass'.\n\n"
            f"Currently registered models:\n{model_list}"
        )
        raise RuntimeError(
            "A model class name or path must be provided. "
            "e.g. 'HyraxCNN' or 'my_package.my_module.MyModelClass'."
        )

    model_cls = cast(type[nn.Module], get_or_load_class(model_name, MODEL_REGISTRY))

    return model_cls
