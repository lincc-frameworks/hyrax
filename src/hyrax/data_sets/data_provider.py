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

    def __init__(self, data_request: dict, config: dict):
        """Initialize the DataProvider with the given data query and a hyrax
        config.

        Parameters
        ----------
        data_request : dict
            A dictionary where keys are dataset names and values are lists of fields
        config : dict
            The Hyrax configuration that can is passed to each dataset instance.
        """
        self.data_request = data_request
        self.config = config
        self.prepped_datasets = {}

    def is_iterable(self):
        """??? Boilerplate code needed for now, maybe not forever ???"""
        return False

    def is_map(self):
        """??? Boilerplate code needed for now, maybe not forever ???"""
        return True

    def prepare_datasets(self):
        """Instantiate each of the requested datasets based on the `data` dictionary,
        and store the instances in the `prepped_datasets` dictionary."""
        for ds in self.data_request:
            if ds not in DATA_SET_REGISTRY:
                logger.error(
                    f"Unable to locate dataset, '{ds}' in the registered datasets:\
                        {list(DATA_SET_REGISTRY.keys())}."
                )
            else:
                self.prepped_datasets[ds] = DATA_SET_REGISTRY[ds](self.config)

            #! ??? Questionable choice here - if object_id is requested and it's
            #! value is truthy, it's magical, and we'll set the object_id at the
            #! top level of the returned data dictionary.
            if "object_id" in self.data_request[ds] and self.data_request[ds]["object_id"]:
                self.primary_dataset = ds

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

    def __getitem__(self, idx):
        """Wrapper that allows this class to be used as a PyTorch Dataset."""
        return self.resolve_data(idx)

    def __len__(self):
        """Returns the length of the dataset based on the first prepared dataset."""
        keys = list(self.prepped_datasets.keys())
        return len(self.prepped_datasets[keys[0]])

    def ids(self):
        """Returns the IDs of the first prepared dataset."""
        keys = list(self.prepped_datasets.keys())
        return self.prepped_datasets[keys[0]].ids()

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
        for ds, fields in self.data_request.items():
            returned_data[ds] = {}
            for field in fields:
                resolved_data = getattr(self.prepped_datasets[ds], f"get_{field}")(idx)
                returned_data[ds][field] = resolved_data
        if self.primary_dataset:
            # If we have a primary dataset, we set the object_id at the top level
            returned_data["object_id"] = returned_data[self.primary_dataset]["object_id"]
        return returned_data
