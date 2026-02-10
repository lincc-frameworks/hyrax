import copy
import logging
import shutil
import warnings
from pathlib import Path

from hyrax.data_sets.data_set_registry import DATASET_REGISTRY

logger = logging.getLogger(__name__)


class DataRequest:
    """DataRequest class to encapsulate split logic validation."""

    def __init__(self, config, required_splits=None, optional_splits=None):
        self.config = config
        self.data_request = self.generate_data_request_from_config(config)
        self.required_splits = required_splits or []
        self.optional_splits = optional_splits or []

        self.provided_splits = set(self.data_request.keys())

    def validate_data_request(self):
        """Validate the data request in the config file. This includes:
        - Pydantic will check that split_percent and split_file are not both defined.
        - Resolve data_location paths and confirm that they exist.
        - Check the split definitions
            - Split percentages defined - check that they sum to 1 if they are
                from the same data location.
            - Split file defined - check that the file exists.
        - Ensure that the required splits are provided in the data request.
        """
        self._check_for_required_splits()
        self._resolve_data_locations()
        self._gather_split_information()
        self._check_split_definitions()

    def create_split_definitions(self, results_dir):
        """Create the split definitions and write them out. This includes:
        1) If no split is defined, create a split definition that uses everything
            in the data directory as the training set.
        2) If split percentages are defined, create a split definition that splits
            the data according to the percentages.
        3) If a split file is defined, read the split file and create a split
            definition based on the contents of the file.

        Parameters
        ----------
        results_dir : str or Path
            The full path to the results sub-directory where split definitions
            should be written.

        Returns
        -------
        dict
            A dictionary where keys are split names and values are paths to the
            split definition files.
        """
        # TODO: Determine what is correct if no split percent or files are defined.
        if not len(self.split_percentages) and not len(self.split_files):
            return {}

        # If split files are defined, we can copy them to the results directory
        if self.split_files:
            split_definition_paths = {}
            for split_name, split_file in self.split_files.items():
                split_file_path = Path(split_file)
                destination_path = Path(results_dir) / split_file_path.name
                shutil.copy2(split_file_path, destination_path)
                split_definition_paths[split_name] = str(destination_path)
            return split_definition_paths

        # If split_percentages are defined, we need to create the split lists
        # and write them out to the results directory.

        # At this point we have to instantiate the dataset to create the splits.
        # TODO: Determine if this is really the right thing to do here. Or if
        # we should just validate the splits and let the splitting happen where
        # it has been happening - later in the dataloader setup.

    @staticmethod
    def generate_data_request_from_config(config):
        """This function handles the backward compatibility issue of defining the
        requested dataset using the deprecated `[model_inputs]` configuration key.

        If neither `[data_request]` nor `[model_inputs]` is defined, an error
        will be raised.

        NOTE: The `[model_inputs]` key is deprecated and will be removed in a
        future version. Users should migrate to using `[data_request]` instead.

        Parameters
        ----------
        config : dict
            The Hyrax configuration that is passed to each dataset instance.

        Returns
        -------
        dict
            A dictionary where keys are dataset names and values are lists of fields

        Raises
        ------
        RuntimeError
            If neither `data_request` nor `model_inputs` is provided in the configuration.
        """

        # Support both 'data_request' (new) and 'model_inputs' (deprecated)
        # Priority: use data_request if it has content, otherwise check model_inputs
        has_data_request = "data_request" in config and config["data_request"]
        has_model_inputs = "model_inputs" in config and config["model_inputs"]

        if has_data_request:
            data_request = copy.deepcopy(config["data_request"])
        elif has_model_inputs:
            warnings.warn(
                "The [model_inputs] configuration key is deprecated and will be "
                "removed in a future version. Please use [data_request] instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data_request = copy.deepcopy(config["model_inputs"])
        elif "data_request" in config or "model_inputs" in config:
            # One of the keys exists but is empty - use the empty dict to trigger error below
            data_request = config.get("data_request") or config.get("model_inputs")
        else:
            # Neither key exists, set empty to trigger error message
            data_request = {}

        # Check if data_request is empty and provide helpful error message
        if not data_request:
            available_datasets = sorted(DATASET_REGISTRY.keys())
            error_msg = """The [data_request] table in your configuration is empty.

    You must provide dataset definitions for training and/or inference:
    - For training: provide "train" and optionally "validate" dataset definitions
    - For inference: provide "infer" dataset definition

    Example configuration:
    [data_request.train]
    [data_request.train.data]
    dataset_class = "HyraxRandomDataset"
    data_location = "./data"
    primary_id_field = "object_id"

    [data_request.infer]
    [data_request.infer.data]
    dataset_class = "HyraxRandomDataset"
    data_location = "./data"
    primary_id_field = "object_id"

    """
            if available_datasets:
                error_msg += "Available built-in dataset classes:\n  - " + "\n  - ".join(available_datasets)
                error_msg += "\n\n"
            error_msg += """For more information and examples, see the documentation at:
    https://hyrax.readthedocs.io/en/latest/notebooks/model_input_1.html"""
            logger.error(error_msg)
            raise RuntimeError(
                "The [data_request] table in the configuration is empty. "
                "Check the preceding error log for details and help."
            )

        return data_request

    def _check_for_required_splits(self):
        """Check that the required splits are provided in the data request."""
        missing_splits = [split for split in self.required_splits if split not in self.provided_splits]
        if missing_splits:
            logger.error(
                f"Missing required splits in data request: {missing_splits}. "
                f"Required splits: {self.required_splits}."
            )
            raise RuntimeError(
                f"Missing required splits in data request: {missing_splits}. "
                f"Check the preceding error log for details and help."
            )

    def _resolve_data_locations(self):
        """Resolve data_location and split_file paths in the data request."""
        path_keys = ["data_location", "split_file"]
        for split_name, split_def in self.data_request.items():
            for friendly_name, friendly_def in split_def.items():
                for path_key in path_keys:
                    if path_key in friendly_def:
                        original_path = Path(friendly_def[path_key])
                        if not original_path.exists():
                            logger.error(
                                f"Resolved {path_key} path for {split_name}.{friendly_name} "
                                f"does not exist: {original_path}"
                            )
                        resolved_path = original_path.expanduser().resolve()
                        friendly_def[path_key] = str(resolved_path)

    def _gather_split_information(self):
        """Gather split information from the data request. Note that we only
        gather split information for the primary dataset in each split, which is
        identified by the presence of the "primary_id_field" key in the data request.

        The split information gathered includes:
        1) Collecting data locations for each split
        2) Collecting split percentages for each split (if defined)
        3) Collecting split files for each split (if defined) and check that they exist
        """
        self.data_locations = {}
        self.split_percentages = {}
        self.split_files = {}
        for split_name, split_def in self.data_request.items():
            for friendly_name, friendly_def in split_def.items():
                # Splits will only be defined for the primary dataset
                if "primary_id_field" in friendly_def:
                    self.data_locations[split_name] = friendly_def["data_location"]

                    # Note - Pydantic validation will prevent prevent split_percent
                    # and split_file from both being defined.
                    if "split_percent" in friendly_def:
                        self.split_percentages[split_name] = friendly_def["split_percent"]
                    elif "split_file" in friendly_def:
                        split_file_path = Path(friendly_def["split_file"])
                        if not split_file_path.exists():
                            logger.error(
                                f"Split file path for {split_name}.{friendly_name} "
                                f"does not exist: {split_file_path}"
                            )
                        self.split_files[split_name] = friendly_def["split_file"]

    def _check_split_definitions(self):
        """Check the split definitions in the data request. This includes:
        - If some split percentages are defined, they are defined for all splits.
        - If some split files are defined, they are defined for all splits.
        - If split percentages defined - check that the sum of percentages for
              splits that share the same data location is <= 1.0.
        """

        # We required that all the provided splits have either 1) No split,
        # 2) split percentages, or 3) split files. i.e. no mixing of split types
        if not len(self.split_percentages) and not len(self.split_files):
            # No splits defined, this is fine - we will use all the data in each provided split.
            return

        # If split_percent was defined for some, but not all, raise an error.
        if len(self.split_percentages) and len(self.split_percentages) != len(self.provided_splits):
            logger.error(
                f"Split percentages are provided for some but not all splits. "
                f"Provided splits: {self.provided_splits}. "
                f"Splits with split percentages: {list(self.split_percentages.keys())}."
            )
            raise RuntimeError(
                "Split percentages are provided for some but not all splits. "
                "Check the preceding error log for details and help."
            )

        # If split_files was defined for some, but not all, raise an error.
        if len(self.split_files) and len(self.split_files) != len(self.provided_splits):
            logger.error(
                f"Split files are provided for some but not all splits. "
                f"Provided splits: {self.provided_splits}. "
                f"Splits with split files: {list(self.split_files.keys())}."
            )
            raise RuntimeError(
                "Split files are provided for some but not all splits. "
                "Check the preceding error log for details and help."
            )

        # For each unique data_location, check that the split percentages sum to <= 1.0
        if self.split_percentages:
            data_location_to_splits = {}
            for split_name, data_location in self.data_locations.items():
                if data_location not in data_location_to_splits:
                    data_location_to_splits[data_location] = []
                data_location_to_splits[data_location].append(split_name)

            for data_location, splits in data_location_to_splits.items():
                total_split_percent = sum(
                    [
                        self.split_percentages[split_name]
                        for split_name in splits
                        if split_name in self.split_percentages
                    ]
                )
                if total_split_percent > 1.0:
                    logger.error(
                        f"Split percentages for data in {data_location} must sum to <= 1.0"
                        f" Currently sums to {total_split_percent}."
                        f" Split percentages: "
                        f"{{split_name: self.split_percentages[split_name] for split_name in splits}}"
                    )
