"""Factory functions for creating result dataset writers and readers.

These factories handle the selection between Lance and .npy formats.
"""

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

LANCE_DB_DIR = "lance_db"


def create_results_writer(result_dir: Union[str, Path]):
    """Create a writer for results (Lance format).

    This factory creates a ResultDatasetWriter for writing inference results
    to Lance format. New writes always use Lance format going forward.

    Parameters
    ----------
    result_dir : Union[str, Path]
        Directory where results should be saved

    Returns
    -------
    ResultDatasetWriter
        Writer instance for Lance storage
    """
    from hyrax.data_sets.result_dataset import ResultDatasetWriter

    return ResultDatasetWriter(result_dir)


def load_results_dataset(
    config: dict, results_dir: Union[Path, str, None] = None, verb: Union[str, None] = None
):
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
    from hyrax.config_utils import resolve_results_dir
    from hyrax.data_sets.inference_dataset import InferenceDataSet
    from hyrax.data_sets.result_dataset import ResultDataset

    # Resolve results directory
    resolved_dir = resolve_results_dir(config, results_dir, verb)

    # Check if Lance format exists
    lance_dir = resolved_dir / LANCE_DB_DIR
    if lance_dir.exists():
        logger.debug(f"Detected Lance format in {results_dir}")
        return ResultDataset(config, resolved_dir)
    else:
        logger.debug(f"Detected .npy format in {results_dir}")
        return InferenceDataSet(config, resolved_dir, verb)
