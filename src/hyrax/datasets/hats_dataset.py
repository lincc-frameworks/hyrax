from pathlib import Path
from types import MethodType

from hyrax.datasets.dataset_registry import HyraxDataset


class HyraxHATSDataset(HyraxDataset):
    """Generic Hyrax dataset for HATS catalogs loaded through LSDB.

    Notes
    -----
    This phase-1 implementation materializes the LSDB catalog to a pandas
    DataFrame and dynamically creates ``get_<column>`` methods for requested columns.
    """

    def __init__(self, config: dict, data_location: Path = None):
        if data_location is None or data_location is False:
            raise ValueError("A `data_location` to a HATS catalog must be provided.")

        self.data_location = data_location
        requested_columns = self._requested_columns_from_config(config)

        import lsdb

        if requested_columns:
            catalog = lsdb.read_hats(data_location, columns=requested_columns)
        else:
            catalog = lsdb.read_hats(data_location)
        self.dataframe = catalog.compute()
        self.column_names = list(self.dataframe.columns)

        def _make_getter(column: str):
            def getter(self, idx: int, _col: str = column):
                import numpy as np
                import pandas as pd

                ret_val = self.dataframe.iloc[idx][_col]
                if isinstance(ret_val, pd.Series):
                    ret_val = ret_val.to_numpy()
                elif isinstance(ret_val, (list, tuple)):
                    ret_val = np.asarray(ret_val)
                return ret_val

            return getter

        for col in self.column_names:
            method_name = f"get_{col}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(col), self))

        super().__init__(config)

    def _requested_columns_from_config(self, config: dict) -> list[str]:
        data_request = config.get("data_request") or config.get("model_inputs") or {}
        requested_columns = set()
        target_location = str(Path(self.data_location).resolve())

        for request_group in data_request.values():
            for dataset_definition in request_group.values():
                if dataset_definition.get("dataset_class") != type(self).__name__:
                    continue
                if str(Path(dataset_definition["data_location"]).resolve()) != target_location:
                    continue

                requested_columns.update(dataset_definition.get("fields", []))

                primary_id_field = dataset_definition.get("primary_id_field")
                if primary_id_field:
                    requested_columns.add(primary_id_field)

                join_field = dataset_definition.get("join_field")
                if join_field:
                    requested_columns.add(join_field)

        return sorted(requested_columns)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Currently required by Hyrax machinery, but likely to be phased out."""
        return {"data": {col: self.dataframe.iloc[idx][col] for col in self.column_names}}

    def sample_data(self):
        """Return the first record in dictionary form as a sample."""
        return {"data": {col: self.dataframe.iloc[0][col] for col in self.column_names}}
