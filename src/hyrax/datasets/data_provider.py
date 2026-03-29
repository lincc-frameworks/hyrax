import copy
import functools
import logging
import time
import warnings
from typing import Any

import numpy as np

from hyrax.datasets.dataset_registry import DATASET_REGISTRY, fetch_dataset_class
from hyrax.tensorboardx_logger import get_tensorboard_logger

logger = logging.getLogger(__name__)
tensorboardx_logger = get_tensorboard_logger()


@functools.singledispatch
def _handle_nans(batch, config):
    """The default _handle_nan function. Will print a warning and return `batch`."""
    logger.warning(
        f"Encountered an unhandled batch type, {type(batch)}, while\
                   attempting to handle NaN values in the data."
    )
    return batch


@_handle_nans.register(np.ndarray)
def _handle_nans_numpy(batch, config):
    return _handle_nans_logic_numpy(batch, config)


# Register tuples and lists for backward compatibility and edge cases.
# NaN handling now primarily occurs in DataProvider.collate() on numpy arrays
# before prepare_data() is called, so tuple/list batches are not expected in
# the main data flow but may still appear from legacy or unusual inputs.
@_handle_nans.register(tuple)
@_handle_nans.register(list)
def _handle_nans_tuple(batch, config):
    """This is the tuple-specific implementation of _handle_nans. Each element
    of the tuple will have nan-handling applied.
    Non-numpy elements are returned unchanged."""

    # Process each element in the tuple
    handled_elements = []
    for element in batch:
        if isinstance(element, np.ndarray):
            handled_elements.append(_handle_nans_logic_numpy(element, config))
        else:
            # Keep non-numpy elements unchanged (e.g., labels, metadata)
            handled_elements.append(element)

    return tuple(handled_elements)


def _handle_nans_logic_numpy(batch, config):
    # Skip non-numeric arrays (e.g., strings, objects)
    if not np.issubdtype(batch.dtype, np.floating):
        return batch

    if config["data_set"]["nan_mode"] is False:
        if np.any(np.isnan(batch)):
            msg = "Input data contains NaN values. This may mean your model output is all NaNs."
            msg += "Consider setting config['data_set']['nan_mode'] = 'quantile' or 'zero' or writing a "
            msg += "to_tensor() function for your model. Search hyrax readthedocs for 'to_tensor' "
            msg += "to get started."
            logger.warning(msg)
        return batch

    if config["data_set"]["nan_mode"] == "quantile":
        quantile = config["data_set"]["nan_quantile"]
        if quantile < 0.0 or quantile > 1.0:
            raise RuntimeError('set config["data_set"]["nan_quantile"] to a value between 0 and 1')
        return _handle_nan_quantile_numpy(batch, quantile)
    elif config["data_set"]["nan_mode"] == "zero":
        return _handle_nan_zero_numpy(batch)
    else:
        msg = f"nan mode was set to '{config['data_set']['nan_mode']}' which is unsupported."
        msg += "The supported modes are 'quantile' and 'zero'."
        raise NotImplementedError(msg)


def _handle_nan_quantile_numpy(batch, quantile):
    if np.any(np.isnan(batch)):
        flat_batch = np.reshape(batch, (batch.shape[0], -1))
        batch_quantile = np.nanquantile(flat_batch, q=quantile, axis=-1)
        for i, val in enumerate(batch_quantile):
            batch[i] = np.nan_to_num(batch[i], nan=val)

    return batch


def _handle_nan_zero_numpy(batch):
    if np.any(np.isnan(batch)):
        batch = np.nan_to_num(batch, nan=0.0)

    return batch


def generate_data_request_from_config(config):
    """This function handles the backward compatibility issue of defining the requested
    dataset using the deprecated `[model_inputs]` configuration key.

    If neither `[data_request]` nor `[model_inputs]` is defined, an error will be raised.

    NOTE: The `[model_inputs]` key is deprecated and will be removed in a future version.
    Users should migrate to using `[data_request]` instead.

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
            "The [model_inputs] configuration key is deprecated and will be removed in a future version. "
            "Please use [data_request] instead.",
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
        # TODO: Update the link when the documentation on data_request is available
        error_msg += """For more information and examples, see the documentation at:
  https://hyrax.readthedocs.io/en/latest/notebooks/model_input_1.html"""
        logger.error(error_msg)
        raise RuntimeError(
            "The [data_request] table in the configuration is empty. "
            "Check the preceding error log for details and help."
        )

    return data_request


class DataProvider:
    """This class presents itself as a PyTorch Dataset, but acts like a GraphQL
    gateway that fetches data from multiple datasets based on the `data_request`
    dictionary provided during initialization.

    This class allows for flexible data retrieval from multiple dataset classes,
    each of which can have different fields requested.

    Additionally, the user can provide specific configuration options for each
    dataset class that will be merged with the original configuration provided
    during initialization.
    """

    def __init__(self, config: dict, request: dict):
        """Initialize the DataProvider with a Hyrax config and extract (or create)
        the data_request.

        Parameters
        ----------
        config : dict
            The Hyrax configuration that defines the data_request.
        request : dict
            A dictionary that defines the data request.
        """

        self.config = config
        self.data_request = request

        self.prepped_datasets = {}
        self.dataset_getters = {}
        self.all_metadata_fields = {}
        self.requested_fields = {}

        # This dictionary maintains a mapping of friendly name to callable collate
        # functions defined on the requested dataset class.
        self.custom_collate_functions = {}

        self.primary_dataset = None
        self.primary_dataset_id_field_name = None
        self.split_fraction = None
        self.primary_data_location = None

        # Assigned externally by setup_dataset after construction when
        # split_fraction-based partitioning is in use.  When set, this
        # contains the list of indices that this provider should serve.
        self.split_indices = None

        # Join support: populated by _build_join_indices after prepare_datasets.
        # Maps friendly_name → join_field name for datasets that use joining.
        self._join_fields: dict[str, str] = {}
        # Maps friendly_name → {str(join_key): secondary_index}.
        self._join_maps: dict[str, dict[str, int]] = {}
        # Ordered list of primary indices that have matches in ALL joined
        # secondaries (inner join).  None when no joins are configured.
        self._joined_primary_indices: list[int] | None = None

        self.prepare_datasets()

        if self.primary_dataset is None or self.primary_dataset_id_field_name is None:
            msg = "No Primary Dataset Defined. Somehow a DataProvider was made without pydantic validation."
            raise RuntimeError(msg)

        if self._join_fields:
            self._build_join_indices()

        self.pull_up_primary_dataset_methods()

        # Required because of circular import.
        from hyrax.datasets.data_cache import DataCache

        self.data_cache = DataCache(config, self)
        self.data_cache.start_preload_thread()

    def pull_up_primary_dataset_methods(self):
        """If a primary dataset is defined, we will pull up some of its methods
        to the DataProvider level so that they can be called directly on the
        DataProvider instance."""

        if self.primary_dataset:
            primary_dataset_instance = self.prepped_datasets[self.primary_dataset]

            # extend this tuple with more prefixes as needed
            exclude_prefixes = ("_", "get_")
            lifted_methods = [
                name
                for name in dir(primary_dataset_instance)
                if not any(name.startswith(p) for p in exclude_prefixes)
                and callable(getattr(primary_dataset_instance, name, None))
            ]

            for method_name in lifted_methods:
                if not hasattr(self, method_name):
                    setattr(self, method_name, getattr(primary_dataset_instance, method_name))

    def __getitem__(self, idx) -> dict:
        """This method returns data for a given index.

        It is also a wrapper that allows this class to be treated as a PyTorch
        Dataset.

        Parameters
        ----------
        idx : int
            The index of the data item to retrieve.

        Returns
        -------
        dict
            A dictionary containing the requested data from the prepared datasets.
        """
        return self.resolve_data(idx)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        When join_field-based joining is active, returns the number of items
        that matched across all joined datasets (inner join count).
        Otherwise, returns the length of the primary (or first) dataset.
        """
        if self._joined_primary_indices is not None:
            return len(self._joined_primary_indices)
        return len(self._primary_or_first_dataset())

    def __repr__(self) -> str:
        repr_str = ""
        for friendly_name, data in self.data_request.items():
            if isinstance(data, dict):
                if self.primary_dataset == friendly_name:
                    repr_str += f"Name: {friendly_name} (primary dataset)\n"
                else:
                    repr_str += f"Name: {friendly_name}\n"
                repr_str += f"  Dataset class: {data['dataset_class']}\n"
                if "data_location" in data:
                    repr_str += f"  Data location: {data['data_location']}\n"
                if "split_fraction" in data:
                    repr_str += f"  Fraction of data to use: {data['split_fraction']}\n"
                if self.primary_dataset_id_field_name:
                    repr_str += f"  Primary ID field: {self.primary_dataset_id_field_name}\n"
                if friendly_name in self._join_fields:
                    repr_str += f"  Join field: {self._join_fields[friendly_name]}\n"
                if "fields" in data:
                    repr_str += f"  Requested fields: {', '.join(data.get('fields', []))}\n"
                else:
                    repr_str += "  Requested fields: *All available fields*\n"
                if "dataset_config" in data:
                    repr_str += "  Dataset config:\n"
                    for k, v in data["dataset_config"].items():
                        repr_str += f"    {k}: {v}\n"
        return repr_str

    def fields(self) -> dict:
        """Print all the available fields for each dataset in the DataProvider.

        Returns
        -------
        dict
            A dictionary mapping friendly dataset names to their available fields.
        """
        fields_dict: dict[str, list[str]] = {}
        for friendly_name, fields in self.dataset_getters.items():
            fields_dict[friendly_name] = list(fields.keys())
        return fields_dict

    def prepare_datasets(self):
        """Instantiate each of the requested datasets based on the ``data_request``
        configuration dictionary. Store the prepared instances in the
        ``self.prepped_datasets`` dictionary."""

        if len(self.data_request) == 0:
            raise RuntimeError("No datasets were requested in `data_request`.")

        for friendly_name, dataset_definition in self.data_request.items():
            dataset_class = dataset_definition.get("dataset_class")
            if not dataset_class:
                logger.error(f"Model input for '{friendly_name}' does not specify a 'dataset_class'.")
                raise RuntimeError(f"Model input for '{friendly_name}' does not specify a 'dataset_class'.")

            # It's ok for data_location to be None, some datasets
            # (e.g. HyraxRandomDataset) may not require it.
            data_location = dataset_definition.get("data_location")

            # Create a temporary config dictionary that merges the original
            # config with the dataset-specific config.
            dataset_specific_config = self._apply_configurations(self.config, dataset_definition)

            # Instantiate the dataset class
            dataset_cls = fetch_dataset_class(dataset_class)
            dataset_instance = dataset_cls(config=dataset_specific_config, data_location=data_location)

            # If the dataset instance has a `collate` method, store it for use in
            # the DataLoader.collate function.
            if hasattr(dataset_instance, "collate") and callable(dataset_instance.collate):
                self.custom_collate_functions[friendly_name] = dataset_instance.collate

            # Store the prepared dataset instance in the `self.prepped_datasets`
            self.prepped_datasets[friendly_name] = dataset_instance

            # If no fields were specifically requested, we'll assume that the user
            # wants _all_ the available fields - user defined and dynamically created!
            if not dataset_definition.get("fields", []):
                dataset_definition["fields"] = [
                    method[4:] for method in dir(dataset_instance) if method.startswith("get_")
                ]

            for field in dataset_definition.get("fields", []):
                if not hasattr(dataset_instance, f"get_{field}"):
                    logger.error(
                        f"No `get_{field}` method for requested field, '{field}' "
                        f"was found in dataset {dataset_class}."
                    )

            # Cache all of the `get_<field_name>` methods in the dataset instance
            # so that we don't have to look them up each time we call `resolve_data`.
            self.dataset_getters[friendly_name] = {}
            for method in dir(dataset_instance):
                if method.startswith("get_"):
                    field_name = method[4:]  # Remove the "get_" prefix
                    self.dataset_getters[friendly_name][field_name] = getattr(dataset_instance, method)

            if len(self.dataset_getters[friendly_name]) == 0:
                logger.error(
                    f"No `get_*` methods were found in the class: {dataset_class}. "
                    "This is likely an error in the dataset class definition."
                )

            # Get all the dataset's metadata fields and store them in
            # `self.all_metadata_fields` dictionary. Modify the name to be
            # <metadata_field_name>_<friendly_name>, i.e. "RA_cifar" or "photoz_hsc".
            if dataset_instance._metadata_table:
                columns = [f"{col}_{friendly_name}" for col in dataset_instance._metadata_table.colnames]
                self.all_metadata_fields[friendly_name] = columns
            else:
                self.all_metadata_fields[friendly_name] = []

            # If this dataset is marked as the primary dataset, store that
            # information for later use.
            if "primary_id_field" in dataset_definition:
                self.primary_dataset = friendly_name
                self.primary_dataset_id_field_name = dataset_definition["primary_id_field"]

                # Store the split_fraction and data_location from the primary
                # dataset's definition.  The Pydantic validator on
                # DataRequestConfig guarantees that split_fraction is only
                # present when primary_id_field is set, so we only need to
                # look for it here.
                self.split_fraction = dataset_definition.get("split_fraction", None)
                self.primary_data_location = dataset_definition.get("data_location", None)

            # Record join_field for secondary datasets that join by key.
            if dataset_definition.get("join_field"):
                self._join_fields[friendly_name] = dataset_definition["join_field"]

            # Cache the requested fields for each dataset as a tuple.
            # Tuples are immutable (preventing accidental modification) and can
            # provide slightly faster iteration than lists, which is beneficial
            # for repeated access in `resolve_data`.
            self.requested_fields[friendly_name] = tuple(dataset_definition.get("fields", []))

    def _build_join_indices(self):
        """Build index mappings for datasets that declare a ``join_field``.

        For each joined secondary dataset, a dict ``{str(key): int(index)}``
        is built by iterating over all items in that dataset.  Then the set
        of primary indices whose IDs appear in **every** joined secondary is
        computed (inner join) and stored in ``_joined_primary_indices``.

        This method is called once during ``__init__`` and is O(N) per
        dataset.  Runtime lookups in ``resolve_data`` are O(1) dict access.
        """
        primary_dataset = self.prepped_datasets[self.primary_dataset]
        primary_id_getter = self.dataset_getters[self.primary_dataset][self.primary_dataset_id_field_name]

        # Build reverse-index for each joined secondary: key_value → idx
        for friendly_name, join_field in self._join_fields.items():
            secondary = self.prepped_datasets[friendly_name]
            getter = self.dataset_getters[friendly_name].get(join_field)
            if getter is None:
                raise RuntimeError(
                    f"Dataset '{friendly_name}' declares join_field='{join_field}' "
                    f"but has no 'get_{join_field}' method."
                )

            reverse_map: dict[str, int] = {}
            n = len(secondary)
            for idx in range(n):
                key = str(getter(idx))
                if key in reverse_map:
                    logger.warning(
                        "Duplicate join key '%s' in dataset '%s' at index %d; "
                        "earlier occurrence at index %d will be shadowed.",
                        key,
                        friendly_name,
                        idx,
                        reverse_map[key],
                    )
                reverse_map[key] = idx
            self._join_maps[friendly_name] = reverse_map

        # Compute inner-join: keep only primary indices present in ALL secondaries.
        valid_primary_indices: list[int] = []
        n_primary = len(primary_dataset)
        for idx in range(n_primary):
            key = str(primary_id_getter(idx))
            if all(key in jm for jm in self._join_maps.values()):
                valid_primary_indices.append(idx)

        if not valid_primary_indices:
            joined_names = ", ".join(self._join_fields.keys())
            raise RuntimeError(
                f"Inner join produced zero matching items between the primary "
                f"dataset '{self.primary_dataset}' and joined datasets "
                f"[{joined_names}]. Verify that the primary_id_field and "
                f"join_field values share common keys."
            )

        n_dropped = n_primary - len(valid_primary_indices)
        if n_dropped > 0:
            logger.info(
                "Dataset join: %d of %d primary items matched across all joined datasets (%d items dropped).",
                len(valid_primary_indices),
                n_primary,
                n_dropped,
            )

        self._joined_primary_indices = valid_primary_indices

    @staticmethod
    def _apply_configurations(base_config: dict, dataset_definition: dict) -> dict:
        """Merge the original base config with the dataset-specific config.

        This function uses ``ConfigManager.merge_configs`` to merge the
        dataset-specific configuration into a copy of the original base config.

        If no ``dataset_config`` is provided in the ``dataset_definition`` dict,
        the original base config will be returned unmodified.

        Data request dictionary examples:

        1) Requesting a built-in Hyrax dataset, "MyDataset"

        .. code-block:: python

            "my_dataset": {
                "dataset_class": "MyDataset",
                "data_location": "/path/to/data",
                "dataset_config": {
                    "MyDataset": {
                        "param1": "value1",
                        "param2": "value2"
                    }
                },
                "fields": ["field1", "field2"]
            }

        or equivalently in a .toml file:

        .. code-block:: toml

            [data_request]
            [data_request.my_dataset]
            dataset_class = "MyDataset"
            data_location = "/path/to/data"
            fields = ["field1", "field2"]
            [data_request.my_dataset.dataset_config.MyDataset]
            param1 = "value1"
            param2 = "value2"

        Here the ``dataset_config`` dictionary will be merged into
        the original base config, overriding the values of param1 and param2
        when creating an instance of ``MyDataset``.

        2) Requesting an external dataset (not built-in), "ExternalDataset"
        Note that the dictionary nesting under "dataset_config" will match the
        dictionary structure in the external dataset's default_config file.

        .. code-block:: python

            "my_dataset": {
                "dataset_class": "ExternalDataset",
                "data_location": "/path/to/data",
                "dataset_config": {
                    "external_example": {
                        "ExternalDataset": {
                            "param1": "value1",
                            "param2": "value2"
                        },
                    },
                },
                "fields": ["field1", "field2"]
            }

        or equivalently in a .toml file:

        .. code-block:: toml

            [data_request]
            [data_request.my_dataset]
            dataset_class = "ExternalDataset"
            data_location = "/path/to/data"
            fields = ["field1", "field2"]
            [data_request.my_dataset.dataset_config.external_example.MyDataset]
            param1 = "value1"
            param2 = "value2"

        Here the ``dataset_config`` dictionary will be merged into
        the original base config, overriding the values of param1 and param2
        when creating an instance of ``ExternalDataset``.

        Parameters
        ----------
        base_config : dict
            The original base configuration dictionary. A copy of this is created,
            the dataset_definition dict is merged into the copy, and the copy
            is returned.

        dataset_definition : dict
            A dictionary defining the dataset, including any dataset-specific
            configuration options in a nested ``dataset_config`` dictionary.

        Returns
        -------
        dict
            A final configuration dictionary to be passed when creating an instance
            of the dataset class.
        """
        from hyrax.config_utils import ConfigManager

        cm = ConfigManager()

        # NOTE: This assumes that the dictionary nesting under dataset_config will
        # either 1) match the built-in dataset class name (e.g. "MyDataset") or
        # 2) match the dictionary structure in the external dataset's default_config
        # file (e.g. "external_example.ExternalDataset").
        if "dataset_config" in dataset_definition:
            tmp_config = {}
            for k, v in dataset_definition["dataset_config"].items():
                if k in DATASET_REGISTRY:
                    tmp_config.setdefault("data_set", {})[k] = v
                else:
                    tmp_config[k] = v

            # Note that `merge_configs` makes a copy of self.config, so the original
            # config will not be modified.
            return cm.merge_configs(base_config, tmp_config)
        else:
            return base_config

    def sample_data(self) -> dict:
        """Returns a data sample. Primarily this will be used for instantiating a
        model so that any runtime resizing can be handled properly.

        Returns
        -------
        dict
            A dictionary containing the data for index 0.
        """
        return self[0]

    def get_object_id(self, idx: int) -> str:
        """Returns the ID at a particular index.

        When joining is active, ``idx`` is a virtual index into the
        joined subset; it is translated to the real primary index first.

        IDs are provided by the primary dataset's primary ID column.
        """
        real_idx = self._joined_primary_indices[idx] if self._joined_primary_indices is not None else idx
        primary_dataset = self.dataset_getters[self.primary_dataset]
        primary_dataset_object_id = primary_dataset[self.primary_dataset_id_field_name](real_idx)
        return str(primary_dataset_object_id)

    def ids(self) -> list[str]:
        """Returns the IDs of the primary dataset.

        Returns
        -------
        list of str
            A list of string IDs corresponding to the primary dataset, ordered by index.
        """
        return [self.get_object_id(idx) for idx in range(len(self))]

    def resolve_data(self, idx: int) -> dict[str, dict[str, Any] | str]:
        """This method requests the field data from the prepared datasets by index.

        When join_field-based joining is active, ``idx`` is a *virtual* index
        into the inner-joined subset.  It is translated to each dataset's
        real index transparently:

        * **Primary dataset** and **non-joined secondaries** receive the real
          primary index (``_joined_primary_indices[idx]``).
        * **Joined secondaries** receive the index looked up from their join
          map using the primary's object ID.

        Parameters
        ----------
        idx : int
            The index of the data item to retrieve.

        Returns
        -------
        dict[str, dict[str, Any] | str]
            A dictionary containing the requested data from the prepared datasets.
            Each key is a dataset friendly name mapped to a dict of field values.
            If a primary dataset is configured, the top-level ``"object_id"`` key
            holds a string representation of the primary ID.
        """
        start_time = time.monotonic_ns()
        prefix = self.__class__.__name__
        cached_data = self.data_cache.try_fetch(idx)
        if cached_data is not None:
            tensorboardx_logger.log_duration_ts(f"{prefix}/cache_hit_s", start_time)
            return cached_data

        # Translate virtual index when joining is active.
        if self._joined_primary_indices is not None:
            primary_idx = self._joined_primary_indices[idx]
            # Pre-fetch the primary object ID for join map lookups.
            primary_id_getter = self.dataset_getters[self.primary_dataset][self.primary_dataset_id_field_name]
            object_id_str = str(primary_id_getter(primary_idx))
        else:
            primary_idx = idx
            object_id_str = None  # computed lazily below if needed

        returned_data: dict[str, dict[str, Any] | str] = {}

        for friendly_name, fields in self.requested_fields.items():
            getters = self.dataset_getters[friendly_name]

            # Determine the real index for this dataset.
            if friendly_name in self._join_maps:
                real_idx = self._join_maps[friendly_name][object_id_str]
            else:
                real_idx = primary_idx

            data_dict = {field: getters[field](real_idx) for field in fields}
            returned_data[friendly_name] = data_dict

        # Because there is machinery in the consuming code that expects an "object_id"
        # key in the returned data, we will add that here if a primary dataset.
        if self.primary_dataset:
            # If the primary id field wasn't already requested, we fetch it now.
            if self.primary_dataset_id_field_name not in returned_data[self.primary_dataset]:
                if object_id_str is not None:
                    object_id = object_id_str
                else:
                    primary_getter = self.dataset_getters[self.primary_dataset]
                    object_id = str(primary_getter[self.primary_dataset_id_field_name](primary_idx))
            else:
                object_id = returned_data[self.primary_dataset][self.primary_dataset_id_field_name]

            returned_data["object_id"] = str(object_id)

        self.data_cache.insert_into_cache(idx, returned_data)
        tensorboardx_logger.log_duration_ts(f"{prefix}/cache_miss_s", start_time)
        return returned_data

    # ^ If we move toward supporting get_<metadata_column_name> methods in datasets,
    # ^ we should be able to remove most or all of this method and the metadata_fields method.
    # ^ This is really here to support the visualization code, and if we convert that
    # ^ to using get_<metadata_column_name> methods, we can remove this.
    # ^ See: https://github.com/lincc-frameworks/hyrax/issues/418

    def metadata(self, idxs=None, fields=None) -> np.ndarray:
        """Fetch the requested metadata fields for the given indices.

        Example:

        .. code-block:: python

            # Fetch the metadata_1 and metadata_2 fields from the dataset with the
            # friendly name "random_1".

            metadata = data_provider.metadata(
                idxs=[0, 1, 2],
                fields=["metadata_1_random_1", "metadata_2_random_1"]
            )

        Parameters
        ----------
        idxs : list of int, optional
            A list of indices for which to fetch metadata. If None, no metadata
            will be returned.
        fields : list of str, optional
            A list of metadata fields to fetch. If None, no metadata will be
            returned.

        Returns
        -------
        np.ndarray
            A structured NumPy array containing the requested metadata fields.
            The dtype names of the array will be the metadata field names, modified
            to include the friendly name of the dataset they come from. For example,
            if the "RA" field comes from a dataset with the friendly name "cifar",
            the returned field name will be "RA_cifar".
        """

        if idxs is None:
            idxs = []

        if fields is None:
            fields = []

        # Create an empty structured array to hold the merged metadata
        returned_metadata = np.empty(0, dtype=[])

        # For each dataset:
        # 1) Find the requested metadata fields that come from it
        # 2) Strip the friendly name from the metadata field name
        # 3) Call the dataset's `metadata` method with indices and metadata fields.
        for friendly_name, dataset in self.prepped_datasets.items():
            metadata_fields_to_fetch = [
                field[: -len(f"_{friendly_name}")] for field in fields if field.endswith(f"_{friendly_name}")
            ]

            if metadata_fields_to_fetch:
                # Translate indices for joined or join-filtered datasets.
                effective_idxs = self._translate_metadata_indices(idxs, friendly_name)
                this_metadata = dataset.metadata(effective_idxs, metadata_fields_to_fetch)
                # Append the friendly name to the columns
                this_metadata.dtype.names = [f"{name}_{friendly_name}" for name in this_metadata.dtype.names]

                # merge this_metadata into the returned_metadata structured array
                if returned_metadata.size == 0:
                    returned_metadata = this_metadata
                else:
                    returned_metadata = np.lib.recfunctions.merge_arrays(
                        (returned_metadata, this_metadata), flatten=True
                    )

        return returned_metadata

    def metadata_fields(self, friendly_name=None) -> list[str]:
        """Returns a list of metadata fields that are available across all prepared
        datasets.

        The field names will be modified to include the friendly name of the
        dataset they come from. For example, if the "RA" field comes from a dataset
        with the friendly name "cifar", the returned field name will be "RA_cifar".

        NOTE: If a specific dataset friendly_name is provided, only the metadata
        fields for that dataset will be returned, and the field names will not
        include the friendly name suffix.

        Parameters
        ----------
        friendly_name : str, optional
            If provided, only the metadata fields for the specified friendly name
            will be returned. If not provided, metadata fields from all datasets
            will be returned.

        Returns
        -------
        list[str]
            The column names of the metadata table passed. Empty list if no metadata
            was provided during construction of the DataProvider.
        """
        all_fields = []
        if friendly_name:
            return [
                field.replace(f"_{friendly_name}", "")
                for field in self.all_metadata_fields.get(friendly_name, [])
            ]

        for _, v in self.all_metadata_fields.items():
            all_fields.extend(v)

        # Always include the `object_id` field
        all_fields.append("object_id")

        return all_fields

    def _translate_metadata_indices(self, idxs, friendly_name):
        """Translate virtual metadata indices to real dataset indices.

        When joining is active, the caller passes *virtual* indices (0-based
        into the joined subset).  This helper converts them to real indices
        for the given dataset, respecting join maps.
        """
        if self._joined_primary_indices is None:
            return idxs

        translated = []
        primary_id_getter = self.dataset_getters[self.primary_dataset][self.primary_dataset_id_field_name]
        for vi in idxs:
            primary_idx = self._joined_primary_indices[vi]
            if friendly_name in self._join_maps:
                key = str(primary_id_getter(primary_idx))
                translated.append(self._join_maps[friendly_name][key])
            else:
                translated.append(primary_idx)
        return translated

    def _primary_or_first_dataset(self):
        """Returns the primary dataset instance if it exists, otherwise returns
        the first dataset in the prepped_datasets."""

        # Get the list of friendly names for the prepared datasets
        keys = list(self.prepped_datasets.keys())

        # If a primary dataset is defined, use that, otherwise use the first one
        dataset_to_use = self.primary_dataset if self.primary_dataset else keys[0]

        return self.prepped_datasets[dataset_to_use]

    def collate(self, batch: list[dict]) -> dict:
        """Custom collate function to be used outside the context of a PyTorch
        DataLoader.

        This function takes a list of data samples (each sample is a dictionary)
        and combines them into a single batch dictionary.

        Parameters
        ----------
        batch : list of dict
            A list of data samples, where each sample is a dictionary.

        Returns
        -------
        dict
            A dictionary where each key corresponds to a field and the value is
            a list of values for that field across the batch.
        """

        batch_dict: dict[str, dict[str, list], list] = {}
        custom_collate: dict[str, list] = {}

        # Aggregate values per friendly_name -> field -> list(values)
        for sample in batch:
            for friendly_name, fields in sample.items():
                # Special handling for "object_id" for the time being. "object_id"
                # hangs on the edge of the data dictionary so that it can be consumed
                # during `infer`, specifically `_save_batch`. Originally it was
                # there to protect against missing ids. We have much more control
                # now with DataProvider, and should remove the special logic for
                # "object_id" from the assorted places it's used.
                if friendly_name == "object_id":
                    val = fields[""] if isinstance(fields, dict) and "" in fields else fields
                    batch_dict.setdefault("object_id", []).append(str(val))
                    continue

                # If we find that `friendly_name` is in self.custom_collate_functions
                # we accumulate the samples from that dataset and hand off to
                # the appropriate custom collate function after the for loop.
                if friendly_name in self.custom_collate_functions:
                    custom_collate.setdefault(friendly_name, []).append(fields)
                    continue

                if friendly_name not in batch_dict:
                    batch_dict[friendly_name] = {}

                for field, value in fields.items():
                    batch_dict[friendly_name].setdefault(field, []).append(value)

        # Convert object_id list -> numpy array of strings
        if "object_id" in batch_dict:
            batch_dict["object_id"] = np.asarray(batch_dict["object_id"], dtype=str)

        # Handle custom collate functions for datasets that define them
        for friendly_name, samples in custom_collate.items():
            # Get the collate function from the mapping dictionary
            custom_collate_fn = self.custom_collate_functions[friendly_name]

            # Pass the list of data samples to the collation
            try:
                custom_collated_data = custom_collate_fn(samples)
            except Exception as err:
                logger.error(
                    f"Error occurred while collating batch for dataset '{friendly_name}' "
                    "using its custom collate function."
                )
                raise RuntimeError(
                    f"Error occurred while collating batch for dataset '{friendly_name}' "
                    "using its custom collate function."
                ) from err

            # Add the collated data to the batch dictionary
            batch_dict[friendly_name] = custom_collated_data

        # Try to convert lists of values into numpy arrays. We skip the "object_id"
        # key since it's already been handled, as well as any keys that are in the
        # self.custom_collate_function dictionary because those should have been
        # handled by the corresponding dataset class custom collate function.
        for friendly_name, fields in batch_dict.items():
            if friendly_name == "object_id":
                continue

            # ! Assuming what is returned from custom_collate is already correctly
            # ! numpy formatted. This is a big assumption. We should provide some
            # ! pre-packaged tests for users developing custom collate functions.
            if friendly_name in self.custom_collate_functions:
                continue

            for field, values in list(fields.items()):
                # If all values are numpy arrays and have identical shapes -> stack
                if all(isinstance(v, np.ndarray) for v in values):
                    shapes = [v.shape for v in values]
                    if all(s == shapes[0] for s in shapes):
                        try:
                            batch_dict[friendly_name][field] = np.stack(values, axis=0)
                            continue
                        except Exception as err:
                            logger.warning(
                                f"Could not stack numpy arrays for field '{field}' "
                                f"in dataset '{friendly_name}'. Consider implementing "
                                "a custom collation function for this dataset."
                            )
                            raise RuntimeError(
                                f"Could not stack numpy arrays for field '{field}' "
                                f"in dataset '{friendly_name}'. Consider implementing "
                                "a custom collation function for this dataset."
                            ) from err
                # if values is a list of numpy scalars convert to numpy array
                if isinstance(values, list):
                    batch_dict[friendly_name][field] = np.array(values)

        # Apply NaN handling to all numpy array fields in the batch,
        # including data produced by custom collate functions.
        for friendly_name, fields in batch_dict.items():
            if friendly_name == "object_id":
                continue

            # Handle dict of fields (normal case)
            if isinstance(fields, dict):
                for field, value in fields.items():
                    if isinstance(value, np.ndarray):
                        batch_dict[friendly_name][field] = _handle_nans(value, self.config)
            # Handle direct numpy arrays (e.g., from custom collate that returns arrays directly)
            elif isinstance(fields, np.ndarray):
                batch_dict[friendly_name] = _handle_nans(fields, self.config)

        return batch_dict
