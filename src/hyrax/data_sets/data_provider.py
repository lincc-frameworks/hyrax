import copy
import logging
from typing import Any

import numpy as np

from hyrax.data_sets.data_set_registry import fetch_dataset_class

logger = logging.getLogger(__name__)


def generate_data_request_from_config(config):
    """This function handles the backward compatibility issue of defining the requested
    dataset in the `[data_set]` table in the config. If a `[model_inputs]` table
    is not defined, we will assemble a `data_request` dictionary from the values
    defined elsewhere in the configuration file.

    NOTE: We should anticipate deprecating the ability to define a data_request in
    `[data_set]`, when that happens, we should be able to remove this function.

    Parameters
    ----------
    config : dict
        The Hyrax configuration that can is passed to each dataset instance.

    Returns
    -------
    dict
        A dictionary where keys are dataset names and values are lists of fields
    """

    if "model_inputs" in config:
        data_request = copy.deepcopy(config["model_inputs"])
    else:
        data_request = {
            "data": {
                "dataset_class": config["data_set"]["name"],
                "data_location": config["general"]["data_dir"],
                "primary_id_field": "object_id",
            },
        }

    return data_request


class DataProvider:
    """This class presents itself as a PyTorch Dataset, but acts like a GraphQL
    gateway that fetches data from multiple datasets based on the `model_inputs`
    dictionary provided during initialization.

    This class allows for flexible data retrieval from multiple dataset classes,
    each of which can have different fields requested.

    Additionally, the user can provide specific configuration options for each
    dataset class that will be merged with the original configuration provided
    during initialization.
    """

    def __init__(self, config: dict):
        """Initialize the DataProvider with a Hyrax config and extract (or create)
        the data_request.

        Parameters
        ----------
        config : dict
            The Hyrax configuration that defines the data_request.
        """

        self.config = config
        self.data_request = generate_data_request_from_config(self.config)

        self.prepped_datasets = {}
        self.dataset_getters = {}
        self.all_metadata_fields = {}
        self.requested_fields = {}

        self.primary_dataset = None
        self.primary_dataset_id_field_name = None

        self.prepare_datasets()

        self.pull_up_primary_dataset_methods()

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
        If the primary dataset is defined, it will return that length, otherwise
        it will use the length of the first dataset in ``self.prepped_datasets``.
        """
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
                if self.primary_dataset_id_field_name:
                    repr_str += f"  Primary ID field: {self.primary_dataset_id_field_name}\n"
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

    def is_iterable(self):
        """DataProvider datasets will always be map-style datasets."""
        return False

    def is_map(self):
        """DataProvider datasets will always be map-style datasets."""
        return True

    def prepare_datasets(self):
        """Instantiate each of the requested datasets based on the ``model_inputs``
        configuration dictionary. Store the prepared instances in the
        ``self.prepped_datasets`` dictionary."""

        if len(self.data_request) == 0:
            raise RuntimeError("No datasets were requested in `model_inputs`.")

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

            # Cache the requested fields for each dataset as a tuple.
            # Tuples are immutable (preventing accidental modification) and can
            # provide slightly faster iteration than lists, which is beneficial
            # for repeated access in `resolve_data`.
            self.requested_fields[friendly_name] = tuple(dataset_definition.get("fields", []))

    @staticmethod
    def _apply_configurations(base_config: dict, dataset_definition: dict) -> dict:
        """Merge the original base config with the dataset-specific config.

        This function uses ``ConfigManager.merge_configs`` to merge the
        dataset-specific configuration into a copy of the original base config.

        If no ``dataset_config`` is provided in the ``dataset_definition`` dict,
        the original base config will be returned unmodified.

        Example of a dataset definition dictionary:

        .. code-block:: python

            "my_dataset": {
                "dataset_class": "MyDataset",
                "data_location": "/path/to/data",
                "dataset_config": {
                    "param1": "value1",
                    "param2": "value2"
                },
                "fields": ["field1", "field2"]
            }

        or equivalently in a .toml file:

        .. code-block:: toml

            [model_inputs]
            [model_inputs.my_dataset]
            dataset_class = "MyDataset"
            data_location = "/path/to/data"
            fields = ["field1", "field2"]
            [model_inputs.my_dataset.dataset_config]
            param1 = "value1"
            param2 = "value2"

        In this example, the ``dataset_config`` dictionary will be merged into
        the original base config, overriding the values of param1 and param2
        when creating an instance of ``MyDataset``.

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

        # NOTE: This assumes that the dataset-specific configuration options
        # are nested under a top-level key that matches the dataset class name.
        # i.e. "data_set": {"MyDataset": {<dataset-specific-options>}}. Or in toml
        # [data_set.MyDataset]
        # <dataset-specific-options>
        # See: https://github.com/lincc-frameworks/hyrax/issues/417

        if "dataset_config" in dataset_definition:
            tmp_config = {
                "data_set": {dataset_definition["dataset_class"]: dataset_definition["dataset_config"]}
            }

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

    # ^ What is the appropriate return when there is no ``ids()`` method in the
    # ^ primary_or_first dataset? Perhaps a generator that yields stop iteration error?
    def ids(self):
        """Returns the IDs of the dataset.

        If the primary dataset is defined it will return those ids, if not,
        it will return the ids of the first dataset in the list of
        prepped_dataset.keys()."""

        primary_dataset = self._primary_or_first_dataset()
        return primary_dataset.ids() if hasattr(primary_dataset, "ids") else []

    def resolve_data(self, idx: int) -> dict:
        """This method requests the field data from the prepared datasets by index.

        Parameters
        ----------
        idx : int
            The index of the data item to retrieve.

        Returns
        -------
        dict
            A dictionary containing the requested data from the prepared datasets.
        """
        returned_data: dict[str, dict[str, Any]] = {}

        for friendly_name, fields in self.requested_fields.items():
            getters = self.dataset_getters[friendly_name]
            data_dict = {field: getters[field](idx) for field in fields}
            returned_data[friendly_name] = data_dict

        # Because there is machinery in the consuming code that expects an "object_id"
        # key in the returned data, we will add that here if a primary dataset.
        if self.primary_dataset:
            # If the primary id field wasn't already requested, we fetch it now.
            if self.primary_dataset_id_field_name not in returned_data[self.primary_dataset]:
                get_fn = self.dataset_getters[self.primary_dataset][self.primary_dataset_id_field_name]
                object_id = get_fn(idx)
            else:
                object_id = returned_data[self.primary_dataset][self.primary_dataset_id_field_name]

            returned_data["object_id"] = object_id

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
                this_metadata = dataset.metadata(idxs, metadata_fields_to_fetch)
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

    def _primary_or_first_dataset(self):
        """Returns the primary dataset instance if it exists, otherwise returns
        the first dataset in the prepped_datasets."""

        # Get the list of friendly names for the prepared datasets
        keys = list(self.prepped_datasets.keys())

        # If a primary dataset is defined, use that, otherwise use the first one
        dataset_to_use = self.primary_dataset if self.primary_dataset else keys[0]

        return self.prepped_datasets[dataset_to_use]
