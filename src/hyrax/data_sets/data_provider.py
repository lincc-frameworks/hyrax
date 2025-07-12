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

    #! Running into a bit of a roadblock - hopefully a byproduct of being tired
    #! The issue at hand is how to tell ``DataProvider`` what configuration to
    #! pay attention to.
    # If this is being used by training or inference, then it makes sense to pay
    # attention to the data_request sent from the model. However, if this is being
    # used to revive a dataset for UMAP or visualization, then it should probably
    # pay attention to the configuration file, and reload from there.
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
        self.all_metadata_fields = {}

        #! A naive stopgap, probably not quite what we need long term. There's
        #! probably a better way to organize the configs here.
        # ? Perhaps an opportunity to break apart the ``[data_set]`` toml table?
        for idx, friendly_name in enumerate(self.data_request):
            self.config[f"dataset_{idx}"] = self.data_request[friendly_name]
            self.config[f"dataset_{idx}"]["friendly_name"] = friendly_name

    def is_iterable(self):
        """??? Boilerplate code needed for now, maybe not forever ???"""
        return False

    def is_map(self):
        """??? Boilerplate code needed for now, maybe not forever ???"""
        return True

    def prepare_datasets(self):
        """Instantiate each of the requested datasets based on the `data` dictionary,
        and store the instances in the `prepped_datasets` dictionary."""
        for friendly_name in self.data_request:
            dataset_definition = self.data_request.get(friendly_name)
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
                self.primary_dataset_id_field = dataset_definition["primary_id_field"]

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
        """Returns the length of the dataset. If the primary dataset is defined
        it will return that, if not, it will default to the first dataset in the
        list of prepped_dataset.keys()."""
        keys = list(self.prepped_datasets.keys())
        k = self.primary_dataset if self.primary_dataset else keys[0]
        return len(self.prepped_datasets[k])

    def ids(self):
        """Returns the IDs of the dataset. If the primary dataset is defined
        it will return those ids, if not, it will return the ids of the first
        dataset in the list of prepped_dataset.keys()."""
        keys = list(self.prepped_datasets.keys())
        k = self.primary_dataset if self.primary_dataset else keys[0]
        return self.prepped_datasets[k].ids()

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
            for field in dataset_definition.get("fields"):
                resolved_data = getattr(self.prepped_datasets[friendly_name], f"get_{field}")(idx)
                returned_data[friendly_name][field] = resolved_data
        if self.primary_dataset:
            returned_data["object_id"] = returned_data[self.primary_dataset][self.primary_dataset_id_field]

        return returned_data

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
        for _, v in self.all_metadata_fields:
            all_fields.extend(v)
        return all_fields
