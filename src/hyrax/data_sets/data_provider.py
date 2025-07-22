import logging

from torch.utils.data import Dataset

from hyrax.data_sets.data_set_registry import DATA_SET_REGISTRY

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

        self.data_request = data_request
        self.config = config
        self.prepped_datasets = {}
        self.all_metadata_fields = {}
        self.iterators = {}  # Only used if this is a set of iterable datasets

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
        """Assume that that the first dataset in prepped datasets is representative
        of all the datasets, and return whether it is an iterable style dataset."""
        pds = self._primary_or_first_dataset()
        return pds.is_iterable()

    def is_map(self):
        """Assume that that the first dataset in prepped datasets is representative
        of all the datasets, and return whether it is a map style dataset."""
        pds = self._primary_or_first_dataset()
        return pds.is_map()

    def metadata(self, idxs=None, fields=None):
        """!!! Boilerplate that doesn't do what we really need it to do !!!"""
        import numpy as np

        shape = (len(idxs), len(fields))
        return np.random.rand(*shape)

    def prepare_datasets(self):
        """Instantiate each of the requested datasets based on the `data` dictionary,
        and store the instances in the `prepped_datasets` dictionary."""
        for friendly_name, dataset_definition in self.data_request.items():
            ds_cls = dataset_definition.get("dataset_class")
            if ds_cls not in DATA_SET_REGISTRY:
                logger.error(
                    f"Unable to locate dataset, '{ds_cls}' in the registered datasets:\
                        {list(DATA_SET_REGISTRY.keys())}."
                )
            else:
                data_directory = dataset_definition.get("data_directory")
                ds_instance = DATA_SET_REGISTRY[ds_cls](self.config, data_directory)
                self.prepped_datasets[friendly_name] = ds_instance

                #! This feels weird - not sure what the right approach is, might
                #! depend on how we move ahead with metadata for visualization.
                if ds_instance._metadata_table:
                    self.all_metadata_fields[friendly_name] = list(ds_instance._metadata_table.colnames)
                else:
                    self.all_metadata_fields[friendly_name] = []

            if "primary_id_field" in dataset_definition:
                self.primary_dataset = friendly_name
                self.primary_dataset_id_field_name = dataset_definition["primary_id_field"]

        is_map = set(ds.is_map() for _, ds in self.prepped_datasets.items())
        if len(is_map) > 1:
            logger.warning(
                "A mixture of map-style and iterable-style datasets were requested. "
                "This behavior is not supported. It is highly recommended to use only one style of dataset."
            )

    def validate_request(self):
        """Convenience method to ensure that each requested dataset exists and that
        each field in each dataset has a `get_<field_name>` method."""
        problem_count = 0
        for ds, fields in self.data_request.items():
            if ds not in DATA_SET_REGISTRY:
                logger.error(
                    f"Unable to locate dataset, '{ds}' in the registered datasets:\
                        {list(DATA_SET_REGISTRY.keys())}."
                )
                problem_count += 1
            for field in fields:
                if not hasattr(DATA_SET_REGISTRY[ds], f"get_{field}"):
                    logger.error(
                        f"No `get_{field}` method for requested field, '{field}' \
                            was found in dataset {ds}."
                    )
                    problem_count += 1

        logger.info(f"Finished validating request. Problems found: {problem_count}")

    def get_sample(self):
        """Returns a data sample. This should dispatch to either __getitem__ or
        __next__ depending on whether the dataset is iterable or map-style."""
        if self.is_iterable():
            return next(iter(self))
        else:
            return self[0]

    def __getitem__(self, idx):
        """Wrapper that allows this class to be used as a PyTorch Dataset."""
        return self.resolve_data_by_index(idx)

    def __iter__(self):
        """Wrapper that allows this class to be used as an IterableDataset."""
        return self.resolve_data_by_iterator()

    def __len__(self):
        """Returns the length of the dataset. If the primary dataset is defined
        it will return that, if not, it will default to the first dataset in the
        list of prepped_dataset.keys()."""
        pds = self._primary_or_first_dataset()
        if pds.is_map():
            # If the dataset is iterable, we can use the length of the iterator
            return len(pds)
        else:
            logger.error("Primary dataset is iterable, cannot determine length.")

    def ids(self):
        """Returns the IDs of the dataset. If the primary dataset is defined
        it will return those ids, if not, it will return the ids of the first
        dataset in the list of prepped_dataset.keys()."""
        pds = self._primary_or_first_dataset()
        return pds.ids() if hasattr(pds, "ids") else []

    def resolve_data_by_index(self, idx):
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
            else:
                # call __getitem__ on the dataset to get all data. Expect that the
                # returned data is a dictionary with a default set of fields
                resolved_data = self.prepped_datasets[friendly_name][idx]
                returned_data[friendly_name] = resolved_data

        return returned_data

    def resolve_data_by_iterator(self):
        """This does the work of requesting the data from the prepared datasets
        with the expectation that all the datasets requested are iterator-style
        datasets.

        Returns
        -------
        dict
            A dictionary containing the requested data from the prepared datasets.
        """
        returned_data = {}
        for friendly_name, ds in self.prepped_datasets.items():
            if friendly_name not in self.iterators:
                self.iterators[friendly_name] = iter(ds)
            returned_data[friendly_name] = next(self.iterators[friendly_name])

        yield returned_data

    #! Same comment here, as in the prepare_datasets method. Not sure if this is
    #! the right approach.
    def metadata_fields(self) -> list[str]:
        """Returns a list of metadata fields supported by this object

        Returns
        -------
        list[str]
            The column names of the metadata table passed. Empty string if no metadata was provided at
            during construction of the HyraxDataset (or derived class).
        """
        all_fields = []
        for _, v in self.all_metadata_fields.items():
            all_fields.extend(v)
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
