import logging
from pathlib import Path
from typing import Any, cast

import torch.nn as nn
from torch import Tensor, as_tensor

from hyrax.plugin_utils import get_or_load_class, update_registry

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def _torch_save(self: nn.Module, save_path: Path):
    import torch

    torch.save(self.state_dict(), save_path)


def _torch_load(self: nn.Module, load_path: Path):
    import torch

    state_dict = torch.load(load_path, weights_only=True)
    self.load_state_dict(state_dict, assign=True)


def _torch_criterion(self: nn.Module):
    """Load the criterion class using the name defined in the config and
    instantiate it with the arguments defined in the config."""

    config = cast(dict[str, Any], self.config)

    # Load the class and get any parameters from the config dictionary
    criterion_name = config["criterion"]["name"]
    criterion_cls = get_or_load_class(criterion_name)

    arguments = {}
    if criterion_name in config:
        arguments = config[criterion_name]

    # Print some debugging info about the criterion function and parameters used
    log_string = f"Using criterion: {criterion_name} "
    if arguments:
        log_string += f"with arguments: {arguments}."
    else:
        log_string += "with default arguments."
    logger.debug(log_string)

    return criterion_cls(**arguments)


def _torch_optimizer(self: nn.Module):
    """Load the optimizer class using the name defined in the config and
    instantiate it with the arguments defined in the config."""

    config = cast(dict[str, Any], self.config)

    # Load the class and get any parameters from the config dictionary
    optimizer_name = config["optimizer"]["name"]
    optimizer_cls = get_or_load_class(optimizer_name)

    arguments = {}
    if optimizer_name in config:
        arguments = config[optimizer_name]

    # Print some debugging info about the optimizer function and parameters used
    log_string = f"Using optimizer: {optimizer_name} "
    if arguments:
        log_string += f"with arguments: {arguments}."
    else:
        log_string += "with default arguments."
    logger.debug(log_string)

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
        cls._criterion = _torch_criterion if not hasattr(cls, "_criterion") else cls._criterion
        cls._optimizer = _torch_optimizer if not hasattr(cls, "_optimizer") else cls._optimizer

    original_init = cls.__init__

    def wrapped_init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        self.criterion = self._criterion()
        self.optimizer = self._optimizer()

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

    model_name = runtime_config["model"]["name"]
    model_cls = None

    if not model_name:
        raise RuntimeError(
            "A model class name or path must be provided. "
            "e.g. 'HyraxCNN' or 'my_package.my_module.MyModelClass'."
        )

    model_cls = cast(type[nn.Module], get_or_load_class(model_name, MODEL_REGISTRY))

    return model_cls
