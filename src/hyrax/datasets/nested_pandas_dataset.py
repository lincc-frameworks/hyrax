from pathlib import Path
from types import MethodType

import pandas as pd

from hyrax.datasets import HyraxDataset


class NestedPandasDataset(HyraxDataset):
    """Hyrax dataset wrapping nested-pandas parquet files."""

    def __init__(self, config: dict, data_location: Path | str | None = None):
        if data_location is None:
            raise ValueError("A `data_location` Path, URL, or directory must be provided.")

        settings = config["data_set"]["NestedPandasDataset"]
        self.data_location = str(data_location)
        self.read_parquet_kwargs = settings["read_parquet_kwargs"]

        self.table = self._load_table()
        self._register_getters()
        super().__init__(config)

    def _load_table(self):
        try:
            import nested_pandas as npd
        except ImportError as err:
            raise ImportError(
                "NestedPandasDataset requires the `nested-pandas` package. "
                "Install it with `pip install nested-pandas`."
            ) from err

        return npd.read_parquet(self.data_location, **self.read_parquet_kwargs)

    def _all_available_fields(self) -> list[str]:
        fields = list(self.table.columns)

        all_columns = getattr(self.table, "all_columns", {})
        for nested_column in getattr(self.table, "nested_columns", []):
            for subcolumn in all_columns[nested_column]:
                fields.append(f"{nested_column}.{subcolumn}")

        return fields

    def _get_value(self, field_name: str, idx: int):
        if "." in field_name:
            nested_column, subcolumn = field_name.split(".", 1)
            return self.table.iloc[int(idx)][nested_column][subcolumn].to_numpy()

        value = self.table.iloc[int(idx)][field_name]
        if isinstance(value, pd.Series):
            return value.to_numpy()
        return value

    def _register_getters(self) -> None:
        def _make_getter(field_name: str):
            def getter(self, idx, _field_name=field_name):
                return self._get_value(_field_name, idx)

            return getter

        for field_name in self._all_available_fields():
            method_name = f"get_{field_name}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(field_name), self))

    def __len__(self) -> int:
        return len(self.table)
