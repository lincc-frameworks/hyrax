from pathlib import Path
from types import MethodType

from hyrax.datasets import HyraxDataset


class HyraxNestedPandasDataset(HyraxDataset):
    """Hyrax dataset for hierarchical data stored in nested-pandas files.

    This class reads parquet files containing nested-pandas dataframes,
    where top-level rows contain sub-dataframes as column values.
    It dynamically creates getter methods for columns from both the
    top-level and nested dataframes.

    Examples
    --------
    Example data_request configuration::

        {
            "train": {
                "data": {
                    "dataset_class": "HyraxNestedPandasDataset",
                    "data_location": "</path/to/data.parquet>",
                    "fields": ["<top_level_col>", "<nested_df>.<nested_col>", ...],
                    "primary_id_field": "<column name that contains a unique ID>",
                },
            },
        }

    Nested column names use dot notation: "nested_table.column_name".
    """

    def __init__(self, config: dict, data_location: Path = None):
        self.data_location = data_location
        if data_location is None:
            raise ValueError("A `data_location` Path to a nested-pandas file must be provided.")

        self.dataframe = self._load_dataframe()
        self.column_names = []

        # Register getters for top-level columns
        for col in self.dataframe.columns:
            clean_name = col.replace(" ", "_").replace("-", "_")
            self.column_names.append(clean_name)
            method_name = f"get_{clean_name}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(self._make_top_level_getter(col), self))

        # Register getters for nested dataframe columns
        # Nested frames are stored in the dataframe and accessed via get_nested_df.column_name
        for nested_frame_col in self.dataframe.columns:
            # Check if this column contains nested dataframes
            sample_value = self.dataframe[nested_frame_col].iloc[0]
            if hasattr(sample_value, "columns"):  # It's a dataframe
                nested_name = nested_frame_col.replace(" ", "_").replace("-", "_")
                for nested_col in sample_value.columns:
                    clean_nested_col = nested_col.replace(" ", "_").replace("-", "_")
                    combined_name = f"{nested_name}.{clean_nested_col}"
                    self.column_names.append(combined_name)
                    method_name = f"get_{nested_name}_{clean_nested_col}"
                    if not hasattr(self, method_name):
                        setattr(
                            self,
                            method_name,
                            MethodType(self._make_nested_getter(nested_frame_col, nested_col), self),
                        )

        super().__init__(config)

    def _load_dataframe(self):
        """Load nested-pandas dataframe from file.

        Supports both pickle and parquet formats. Pickle is preferred for
        nested dataframes as parquet does not support object columns.
        """
        import pandas as pd

        data_location_str = str(self.data_location)
        if data_location_str.endswith(".pkl") or data_location_str.endswith(".pickle"):
            return pd.read_pickle(self.data_location)
        else:
            # Try parquet, but only for non-nested cases
            import nested_pandas as npd

            return npd.read_parquet(self.data_location)

    def _make_top_level_getter(self, column):
        """Create a getter for a top-level dataframe column."""

        def getter(self, idx, _col=column):
            ret_val = self.dataframe[_col].iloc[idx]
            return ret_val

        return getter

    def _make_nested_getter(self, nested_frame_col, nested_col):
        """Create a getter for a column within a nested dataframe."""

        def getter(self, idx, _frame_col=nested_frame_col, _col=nested_col):
            nested_df = self.dataframe[_frame_col].iloc[idx]
            ret_val = nested_df[_col].values
            return ret_val

        return getter

    def __getitem__(self, idx):
        """Currently required by Hyrax machinery, but likely to be phased out."""
        return {}

    def __len__(self) -> int:
        """Return the number of records in the top-level dataframe."""
        return len(self.dataframe)

    def sample_data(self):
        """Return the first record in dictionary form as the sample."""
        sample = {"data": {}}

        for col in self.dataframe.columns:
            clean_name = col.replace(" ", "_").replace("-", "_")
            sample["data"][clean_name] = self.dataframe[col].iloc[0]

        return sample
