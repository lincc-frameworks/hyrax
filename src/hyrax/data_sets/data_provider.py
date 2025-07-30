import logging

import numpy as np
from torch.utils.data import Dataset

from hyrax.data_sets.data_set_registry import DATA_SET_REGISTRY

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_data_request_from_config(config):
    """This function handles the backward compatibility issue of defining the requested
    dataset in the `[data_set]` table in the config. If a `[model_data]` table
    is not defined, we will assemble a data_request dictionary from the values
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

    if "model_data" in config:
        data_request = config["model_data"]
    else:
        # Assume that we want only one dataset, and that the `model_data` table
        # is not present in the config. We need to assemble a data_request
        # based on config['data_set'].
        data_request = {
            "data": {
                "dataset_class": config["data_set"]["name"],
                "data_directory": config["general"]["data_dir"],
                "primary_id_field": "object_id",
            }
        }

    return data_request


class DataProvider(Dataset):
    """This class presents itself as a PyTorch Dataset, but acts like a GraphQL
    gateway that fetches data from multiple datasets based on the `data` dictionary
    provided during initialization. It allows for flexible data retrieval from
    multiple datasets, each of which can have different fields requested.
    """

    def __init__(self, config: dict):
        """Initialize the DataProvider with the given data query and a hyrax
        config.

        Parameters
        ----------
        data_request : dict
            A dictionary where keys are dataset names and values are lists of fields
        config : dict
            The Hyrax configuration that can is passed to each dataset instance.
        """

        self.config = config
        self.data_request = generate_data_request_from_config(self.config)

        self.validate_request()

        self.prepped_datasets = {}
        self.all_metadata_fields = {}

        self.primary_dataset = None
        self.primary_dataset_id_field_name = None

    def __repr__(self):
        repr_str = ""
        for friendly_name, data in self.data_request.items():
            if isinstance(data, dict):
                repr_str += f"{friendly_name}\n"
                repr_str += f"  Dataset class: {data['dataset_class']}\n"
                repr_str += f"  Data directory: {data['data_directory']}\n"
                if "fields" in data:
                    repr_str += f"  Fields: {', '.join(data.get('fields', []))}\n"
                else:
                    repr_str += "  Fields: *All available fields*\n"
        return repr_str

    def is_iterable(self):
        """DataProvider datasets will always be map-style datasets."""
        return False

    def is_map(self):
        """DataProvider datasets will always be map-style datasets."""
        return True

    def metadata(self, idxs=None, fields=None):
        """Fetch metadata for the requested fields and indices."""

        # Create an empty structured array to hold the merged metadata
        returned_metadata = np.empty(0, dtype=[])

        # For each dataset, find the fields that were requested that come from it,
        # strip the friendly name from the field name, and then
        # call the dataset's metadata method with the stripped field names.
        for friendly_name, dataset in self.prepped_datasets.items():
            fetch_fields = [
                field.replace(f"_{friendly_name}", "")
                for field in fields
                if field.endswith(f"_{friendly_name}")
            ]

            if fetch_fields:
                this_metadata = dataset.metadata(idxs, fetch_fields)
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

    def prepare_datasets(self):
        """Instantiate each of the requested datasets based on the `data` dictionary,
        and store the instances in the `prepped_datasets` dictionary."""
        for friendly_name, dataset_definition in self.data_request.items():
            dataset_class = dataset_definition.get("dataset_class")
            if dataset_class not in DATA_SET_REGISTRY:
                logger.error(
                    f"Unable to locate dataset, '{dataset_class}' in the registered datasets:\
                        {list(DATA_SET_REGISTRY.keys())}."
                )
            else:
                data_directory = dataset_definition.get("data_directory")
                dataset_instance = DATA_SET_REGISTRY[dataset_class](self.config, data_directory)

                # If a user creates a DataProvider instance manually, this will
                # guard against iterable-style dataset creeping in. Under normal
                # circumstances iterable-style dataset would be caught prior to this.
                if dataset_instance.is_iterable():
                    logger.error(
                        f"Dataset '{dataset_class}' is an iterable-style dataset. "
                        "This is not supported in the current implementation of DataProvider. "
                        "Hyrax DataProvider only supports 1-N map-style datasets at this time. "
                        "You should instantiate an iterable-style dataset class directly."
                    )

                self.prepped_datasets[friendly_name] = dataset_instance

                # Get all of the column names for a dataset's metadata table and
                # store them in the all_metadata_fields dictionary.
                # Modify the name to be <field_name>_<friendly_name>, i.e. "RA_cifar".
                if dataset_instance._metadata_table:
                    columns = [f"{col}_{friendly_name}" for col in dataset_instance._metadata_table.colnames]
                    self.all_metadata_fields[friendly_name] = columns
                else:
                    self.all_metadata_fields[friendly_name] = []

            if "primary_id_field" in dataset_definition:
                self.primary_dataset = friendly_name
                self.primary_dataset_id_field_name = dataset_definition["primary_id_field"]

    def validate_request(self):
        """Convenience method to ensure that each requested dataset exists and that
        each field in each dataset has a `get_<field_name>` method."""
        problem_count = 0
        for _, dataset_parameters in self.data_request.items():
            dataset_class = dataset_parameters.get("dataset_class")
            if dataset_class not in DATA_SET_REGISTRY:
                logger.error(
                    f"Unable to locate dataset, '{dataset_class}' in the registered datasets:"
                    f" {list(DATA_SET_REGISTRY.keys())}."
                )
                problem_count += 1
            if DATA_SET_REGISTRY[dataset_class].is_iterable():
                logger.error(
                    f"Dataset '{dataset_class}' is an iterable-style dataset. "
                    "This is not supported in the current implementation of DataProvider. "
                    "Hyrax DataProvider only supports 1-N map-style datasets at this time. "
                    "You should instantiate an iterable-style dataset class directly."
                )
                problem_count += 1
            # If "fields" wasn't provided or it's empty or None, attempt to gather
            # all available get_* methods in the dataset class.
            if "fields" not in dataset_parameters or not dataset_parameters["fields"]:
                # Gather all available fields from the dataset class
                dataset_parameters["fields"] = [
                    method[4:]
                    for method in dir(DATA_SET_REGISTRY[dataset_class])
                    if method.startswith("get_")
                ]
                if not dataset_parameters["fields"]:
                    logger.error(
                        f"No fields were found in dataset {dataset_class}. "
                        "This is likely an error in the dataset class definition."
                    )
                    problem_count += 1
            else:
                for field in dataset_parameters.get("fields", []):
                    if not hasattr(DATA_SET_REGISTRY[dataset_class], f"get_{field}"):
                        logger.error(
                            f"No `get_{field}` method for requested field, '{field}' "
                            f"was found in dataset {dataset_class}."
                        )
                        problem_count += 1

        if problem_count > 0:
            logger.error(f"Finished validating request. Problems found: {problem_count}")
            raise RuntimeError("Data request validation failed. See logs for details.")

    def sample_data(self):
        """Returns a data sample. Primarily this will be used for instantiating a
        model so that any runtime resizing can be handled properly."""
        return self[0]

    def __getitem__(self, idx):
        """Wrapper that allows this class to be used as a PyTorch Dataset."""
        return self.resolve_data(idx)

    def __len__(self):
        """Returns the length of the dataset. If the primary dataset is defined
        it will return that, if not, it will default to the first dataset in the
        list of prepped_dataset.keys()."""
        return len(self._primary_or_first_dataset())

    def ids(self):
        """Returns the IDs of the dataset. If the primary dataset is defined
        it will return those ids, if not, it will return the ids of the first
        dataset in the list of prepped_dataset.keys()."""
        primary_dataset = self._primary_or_first_dataset()
        return primary_dataset.ids() if hasattr(primary_dataset, "ids") else []

    def resolve_data(self, idx):
        """This does the work of requesting the data from the prepared datasets.

        Parameters
        ----------
        idx : int
            The index of the data item to retrieve.

        Returns
        -------
        dict
            A dictionary containing the requested data from the prepared datasets.
        """
        returned_data = {}
        for friendly_name in self.data_request:
            returned_data[friendly_name] = {}
            dataset_definition = self.data_request.get(friendly_name)
            if dataset_definition.get("fields"):
                for field in dataset_definition.get("fields"):
                    resolved_data = getattr(self.prepped_datasets[friendly_name], f"get_{field}")(idx)
                    returned_data[friendly_name][field] = resolved_data

        if self.primary_dataset:
            returned_data["object_id"] = returned_data[self.primary_dataset][
                self.primary_dataset_id_field_name
            ]

        return returned_data

    def metadata_fields(self, friendly_name=None) -> list[str]:
        """Returns a list of metadata fields supported by this object

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
        all_fields.append("object_id")  # Always include the object_id field
        return all_fields

    def _primary_or_first_dataset(self):
        """Returns the primary dataset instance if it exists, otherwise returns
        the first dataset in the prepped_datasets."""
        keys = list(self.prepped_datasets.keys())
        return (
            self.prepped_datasets[self.primary_dataset]
            if self.primary_dataset
            else self.prepped_datasets[keys[0]]
        )
