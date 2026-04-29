from pathlib import Path
from types import MethodType

from hyrax.datasets.dataset_registry import HyraxDataset


class NestedPandasDataset(HyraxDataset):
    """Hyrax dataset for `nested-pandas <https://nested-pandas.readthedocs.io/>`_ parquet files.

    Notes
    -----
    The file is read once at construction time with ``nested_pandas.read_parquet``.
    A ``get_<column>`` method is created for every top-level column in the
    NestedFrame, including nested columns.

    For a base (scalar) column, ``get_<column>(idx)`` returns the underlying
    Python/NumPy scalar from the row.

    For a nested column, ``get_<column>(idx)`` returns a ``dict`` mapping each
    sub-column name to a NumPy array containing the values for that one object.

    Examples
    --------
    Example data_request configuration::

        {
            "train": {
                "data": {
                    "dataset_class": "NestedPandasDataset",
                    "data_location": "/path/to/file.parquet",
                    "fields": ["flux", "lightcurve"],
                    "primary_id_field": "object_id",
                },
            },
        }
    """

    def __init__(self, config: dict, data_location: Path = None):
        if data_location is None or data_location is False:
            raise ValueError("A `data_location` to a nested-pandas parquet file must be provided.")

        self.data_location = data_location
        settings = config["data_set"][type(self).__name__]
        read_parquet_kwargs = settings["read_parquet_kwargs"]

        from nested_pandas import read_parquet

        self.dataframe = read_parquet(data_location, **read_parquet_kwargs)
        self.column_names = list(self.dataframe.columns)

        def _make_getter(column: str):
            def getter(self, idx: int, _col: str = column):
                import numpy as np
                import pandas as pd

                ret_val = self.dataframe.iloc[idx][_col]
                if isinstance(ret_val, pd.DataFrame):
                    return {sub_col: ret_val[sub_col].to_numpy() for sub_col in ret_val.columns}
                if isinstance(ret_val, pd.Series):
                    return ret_val.to_numpy()
                if isinstance(ret_val, (list, tuple)):
                    return np.asarray(ret_val)
                return ret_val

            return getter

        for col in self.column_names:
            method_name = f"get_{col}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(col), self))

        super().__init__(config)

    def __len__(self) -> int:
        return len(self.dataframe)
