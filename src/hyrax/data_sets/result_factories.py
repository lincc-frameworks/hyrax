"""Factory functions for creating result dataset writers and readers.

These factories handle the selection between Lance and .npy formats.
"""

import logging
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

LANCE_DB_DIR = "lance_db"


def _resolve_results_dir(config: dict, results_dir: Union[Path, str, None], verb: Union[str, None]) -> Path:
    """Resolve the results directory path.

    This helper handles auto-discovery of the most recent results directory if not provided.

    Parameters
    ----------
    config : dict
        The hyrax config dictionary
    results_dir : Union[Path, str, None]
        The results subdirectory to load from
    verb : Union[str, None]
        The name of the verb that generated the results (for auto-discovery)

    Returns
    -------
    Path
        Resolved path to results directory

    Raises
    ------
    RuntimeError
        If results directory cannot be found or does not exist
    """
    from hyrax.config_utils import find_most_recent_results_dir

    verb = "infer" if verb is None else verb

    if results_dir is None:
        if config["results"]["inference_dir"]:
            results_dir = config["results"]["inference_dir"]
            if not isinstance(results_dir, str):
                raise RuntimeError("Configured [results_dir] is not a string")
        else:
            results_dir = find_most_recent_results_dir(config, verb=verb)
            if results_dir is None:
                msg = "Could not find a results directory. Run infer or use "
                msg += "[results] inference_dir config to specify a directory."
                raise RuntimeError(msg)
            msg = f"Using most recent results dir {results_dir} for lookup."
            msg += " Use the [results] inference_dir config to set a directory or pass it to this verb."
            logger.debug(msg)

    retval = Path(results_dir) if isinstance(results_dir, str) else results_dir

    if not retval.exists():
        raise RuntimeError(f"Inference directory {results_dir} does not exist")

    return retval


def create_results_writer(original_dataset: Dataset, result_dir: Union[str, Path]):
    """Create a writer for results (Lance format).

    This factory creates a ResultDatasetWriter for writing inference results
    to Lance format. New writes always use Lance format going forward.

    Parameters
    ----------
    original_dataset : Dataset
        The dataset being processed (currently unused, kept for API compatibility)
    result_dir : Union[str, Path]
        Directory where results should be saved

    Returns
    -------
    ResultDatasetWriter
        Writer instance for Lance storage
    """
    from hyrax.data_sets.result_dataset import ResultDatasetWriter

    # ResultDatasetWriter doesn't need the original dataset, just the directory
    return ResultDatasetWriter(result_dir)


def load_results_dataset(config: dict, results_dir: Union[Path, str, None] = None, verb: Union[str, None] = None):
    """Load a results dataset, auto-detecting format.

    This factory auto-detects whether the results are in Lance or .npy format
    and returns the appropriate dataset class.

    Parameters
    ----------
    config : dict
        The hyrax config dictionary
    results_dir : Union[Path, str, None], optional
        The results subdirectory to load from
    verb : Union[str, None], optional
        The name of the verb that generated the results (for auto-discovery)

    Returns
    -------
    Union[ResultDataset, InferenceDataSet]
        The appropriate dataset instance based on detected format
    """
    from hyrax.data_sets.inference_dataset import InferenceDataSet
    from hyrax.data_sets.result_dataset import ResultDataset

    # Resolve results directory
    resolved_dir = _resolve_results_dir(config, results_dir, verb)

    # Check if Lance format exists
    lance_dir = resolved_dir / LANCE_DB_DIR
    if lance_dir.exists():
        logger.debug(f"Detected Lance format in {results_dir}")
        return ResultDataset(config, resolved_dir)
    else:
        logger.debug(f"Detected .npy format in {results_dir}")
        return InferenceDataSet(config, resolved_dir, verb)
