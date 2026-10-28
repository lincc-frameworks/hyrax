"""Utility functions for model operations."""

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def load_model_weights(config: dict, model, verb: str):
    """Load model weights from the file indicated by the configuration
    or from the most recent training run if no file is specified.

    This function updates the config dict to track which weights file was actually used.

    Parameters
    ----------
    config : dict
        Full runtime configuration
    model : nn.Module
        The model class to load weights into
    verb : str
        The verb name (e.g., "infer", "test") for finding weights file in config

    Raises
    ------
    RuntimeError
        If the weights file cannot be found or does not load properly
    """
    from hyrax.config_utils import find_most_recent_results_dir

    weights_file: Union[str, Path] | None = (
        config[verb]["model_weights_file"] if config[verb]["model_weights_file"] else None
    )

    if weights_file is None:
        recent_results_path = find_most_recent_results_dir(config, "train")
        if recent_results_path is None:
            raise RuntimeError(f"Must define model_weights_file in the [{verb}] section of hyrax config.")

        weights_file = recent_results_path / config["train"]["weights_filename"]

    # Ensure weights file is a path object.
    weights_file_path = Path(weights_file)

    if not weights_file_path.exists():
        raise RuntimeError(f"Model Weights file {weights_file_path} does not exist")

    try:
        model.load(weights_file_path)
        # Update config to track which weights file was actually used
        config[verb]["model_weights_file"] = str(weights_file_path)
        msg = f"Updated config['{verb}']['model_weights_file'] to: "
        msg += f"{config[verb]['model_weights_file']}"
        logger.info(msg)
    except Exception as err:
        msg = f"Model weights file {weights_file_path} did not load properly. Are you sure you are "
        msg += f"{verb}ing with the correct model?"
        raise RuntimeError(msg) from err


def read_input_shape(config: dict, model_name: str) -> tuple[int, int, int]:
    """Read and validate ``[model.<model_name>].input_shape`` from the configuration.

    Image models size their layers from this key at construction, so it has to describe
    one input image completely: ``[channels, height, width]``.

    Parameters
    ----------
    config : dict
        Full runtime configuration
    model_name : str
        Name of the model's config subtable, e.g. ``"HyraxCNN"``

    Returns
    -------
    tuple[int, int, int]
        ``(channels, height, width)``

    Raises
    ------
    ValueError
        If the key is missing or is not a list of three positive integers.
    """
    key = f"config['model']['{model_name}']['input_shape']"
    shape = config["model"][model_name].get("input_shape")

    # `false` is the TOML "not set" sentinel, and bool is a subclass of int, so booleans
    # must be rejected explicitly or `[3, true, 32]` would pass as `[3, 1, 32]`.
    def _is_positive_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    if not isinstance(shape, (list, tuple)) or len(shape) != 3 or not all(map(_is_positive_int, shape)):
        raise ValueError(
            f"{key} must be [channels, height, width], three positive integers describing one "
            f"input image, got {shape!r}. It has to match what the dataset produces (see "
            "[data_set].crop_to)."
        )

    return int(shape[0]), int(shape[1]), int(shape[2])
