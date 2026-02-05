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
