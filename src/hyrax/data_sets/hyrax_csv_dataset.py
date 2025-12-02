import re
from pathlib import Path
from types import MethodType

import pandas as pd

from hyrax.data_sets.data_set_registry import HyraxDataset


class HyraxCSVDataset(HyraxDataset):
    """A Hyrax Dataset for CSV files.
    This class reads a CSV file using pandas with memory mapping enabled.
    It dynamically creates getter methods for each column in the CSV file,
    allowing users to request data from specific columns.

    Note: Column names found in the CSV file are used to create the getter methods.
    If a column name contains characters that are invalid for method names,
    those characters are replaced with underscores.

    Example model_inputs configuration:
    {
        "train": {
            "data": {
                "dataset_class": "HyraxCSVDataset",
                "data_location": </path/to/data.csv>,
                "fields": ["<column1>", "<column2>", ...],
                "primary_id_field": <column name that contains a unique ID>,
            },
        },
        "validate": { <similar to above> },
        "infer": { <similar to above> },
    }
    """

    def __init__(self, config: dict, data_location: Path = None):
        self.data_location = data_location
        if data_location is None:
            raise ValueError("A `data_location` Path to a .csv file must be provided.")

        header_only = pd.read_csv(data_location, nrows=0)
        self.column_names = [re.sub(r"\W", "_", col) for col in list(header_only.columns)]
        self.mem_mapped_csv = pd.read_csv(data_location, memory_map=True, header=0)

        # Automatically generate all the getter methods based on the column names.
        def _make_getter(column):
            def getter(self, idx, _col=column):
                ret_val = self.mem_mapped_csv[_col][idx]
                if isinstance(ret_val, pd.Series):
                    ret_val = ret_val.to_list()
                return ret_val

            return getter

        for col in self.column_names:
            method_name = f"get_{col}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(col), self))

        super().__init__(config)

    def __getitem__(self, idx):
        """Currently required by Hyrax machinery, but likely to be phased out."""
        return {}

    def __len__(self) -> int:
        """Return the number of records in the CSV."""
        return len(self.mem_mapped_csv)

    def sample_data(self):
        """Return the first record, in dictionary form, as the sample."""
        sample = {"data": {}}

        for col in self.column_names:
            sample["data"][col] = self.mem_mapped_csv.iloc[0][col]

        return sample

    @classmethod
    def is_map(cls) -> bool:
        """Boilerplate method to indicate this is a map-style dataset."""
        return True
