import functools
import logging
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch.nn as nn

from hyrax.plugin_utils import (
    get_or_load_class,
    load_prepare_inputs,
    load_to_tensor,
    save_prepare_inputs,
    save_to_tensor,
    update_registry,
)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def deprecated(reason: str):
    """Decorator to mark functions as deprecated.

    Parameters
    ----------
    reason : str
        The reason for deprecation and guidance for what to use instead.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _torch_save(self: nn.Module, save_path: Path):
    import torch

    # save the model weights
    torch.save(self.state_dict(), save_path)
    # Save prepare_inputs (preferred) or fall back to to_tensor for backward compatibility
    if hasattr(self, "prepare_inputs"):
        save_prepare_inputs(self.prepare_inputs, save_path)
    else:
        # For backward compatibility, save to_tensor if prepare_inputs is not available
        save_to_tensor(self.to_tensor, save_path)


def _torch_load(self: nn.Module, load_path: Path):
    import ignite.distributed as idist
    import torch

    # Use ignite's device detection which handles distributed training and device availability
    # This allows models trained on GPU to be loaded on CPU-only machines
    device = idist.device()
    state = torch.load(load_path, weights_only=True, map_location=device)

    self.load_state_dict(state, assign=True)

    # Try loading prepare_inputs first (new name), fall back to to_tensor for backward compatibility
    prepare_inputs_fn = load_prepare_inputs(load_path.parent)
    to_tensor_fn = load_to_tensor(load_path.parent)

    if prepare_inputs_fn:
        if isinstance(prepare_inputs_fn, staticmethod):
            self.prepare_inputs = prepare_inputs_fn
            prepare_inputs_func = prepare_inputs_fn.__func__
        else:
            self.prepare_inputs = staticmethod(prepare_inputs_fn)
            prepare_inputs_func = prepare_inputs_fn

        # Create deprecated wrapper for to_tensor
        @deprecated("Use prepare_inputs instead.")
        def to_tensor_wrapper(data_dict):
            return prepare_inputs_func(data_dict)

        self.to_tensor = staticmethod(to_tensor_wrapper)

    elif to_tensor_fn:
        # Backward compatibility: if only to_tensor is found, use it for both
        logger.warning(
            f"Found to_tensor function in {load_path.parent}. "
            "to_tensor is deprecated, please use prepare_inputs instead."
        )
        if isinstance(to_tensor_fn, staticmethod):
            self.prepare_inputs = to_tensor_fn
            self.to_tensor = to_tensor_fn
        else:
            self.prepare_inputs = staticmethod(to_tensor_fn)
            self.to_tensor = staticmethod(to_tensor_fn)
    else:
        logger.warning(
            f"Could not find prepare_inputs or to_tensor function in {load_path}. "
            "Using the model's existing methods."
        )


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

    def default_prepare_inputs(data_dict):
        if "data" not in data_dict:
            msg = "Hyrax couldn't find a 'data' key in the data dictionaries from your dataset.\n"
            msg += f"We recommend you implement a function on {cls.__name__} to unpack the appropriate\n"
            msg += "value(s) from the dictionary your dataset is returning:\n\n"
            msg += f"class {cls.__name__}:\n\n"
            msg += "    @staticmethod\n"
            msg += "    def prepare_inputs(data_dict) -> Tuple[npt.NDArray, ...]:\n"
            msg += "        <Your implementation goes here>\n\n"
            raise RuntimeError(msg)

        data = data_dict.get("data")
        image = data.get("image", np.array([]))
        label = data.get("label", np.array([]))

        return (image, label)

    # Check if the class has prepare_inputs or to_tensor
    has_prepare_inputs = hasattr(cls, "prepare_inputs")
    has_to_tensor = hasattr(cls, "to_tensor")

    if has_prepare_inputs:
        # Model has new prepare_inputs method - ensure it's a staticmethod
        if not isinstance(vars(cls)["prepare_inputs"], staticmethod):
            msg = f"You must implement prepare_inputs() in {cls.__name__} as\n\n"
            msg += "@staticmethod\n"
            msg += "def prepare_inputs(data_dict: dict) -> torch.Tensor:\n"
            msg += "    <Your implementation goes here>\n"
            raise RuntimeError(msg)

        # Create deprecated to_tensor wrapper that calls prepare_inputs
        # Capture the prepare_inputs method in the closure
        prepare_inputs_func = cls.prepare_inputs.__func__

        @staticmethod
        @deprecated("Use prepare_inputs instead.")
        def deprecated_to_tensor(data_dict):
            return prepare_inputs_func(data_dict)

        cls.to_tensor = deprecated_to_tensor

    elif has_to_tensor:
        # Model only has old to_tensor method - create prepare_inputs that calls it
        # and mark to_tensor as deprecated
        if not isinstance(vars(cls)["to_tensor"], staticmethod):
            msg = f"You must implement to_tensor() in {cls.__name__} as\n\n"
            msg += "@staticmethod\n"
            msg += "def to_tensor(data_dict: dict) -> torch.Tensor:\n"
            msg += "    <Your implementation goes here>\n"
            raise RuntimeError(msg)

        # Wrap the existing to_tensor with deprecation warning
        original_to_tensor = cls.to_tensor.__func__

        @staticmethod
        @deprecated("Use prepare_inputs instead.")
        def deprecated_to_tensor_wrapper(data_dict):
            return original_to_tensor(data_dict)

        cls.to_tensor = deprecated_to_tensor_wrapper

        # Create prepare_inputs that calls the original implementation
        @staticmethod
        def prepare_inputs_from_to_tensor(data_dict):
            return original_to_tensor(data_dict)

        cls.prepare_inputs = prepare_inputs_from_to_tensor

    else:
        # No method defined - use defaults
        cls.prepare_inputs = staticmethod(default_prepare_inputs)

        # Create deprecated to_tensor wrapper
        # Capture default_prepare_inputs in the closure
        default_func = default_prepare_inputs

        @staticmethod
        @deprecated("Use prepare_inputs instead.")
        def deprecated_to_tensor_default(data_dict):
            return default_func(data_dict)

        cls.to_tensor = deprecated_to_tensor_default

    # Update required methods to include prepare_inputs
    required_methods = ["train_step", "forward", "__init__", "prepare_inputs"]
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
