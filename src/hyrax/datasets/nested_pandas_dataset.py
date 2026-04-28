from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd

from hyrax.datasets import HyraxDataset


class NestedPandasDataset(HyraxDataset):
    """Hyrax dataset for nested-pandas parquet files."""

    def __init__(self, config: dict, data_location: Path | str | None = None):
        if data_location is None:
            raise ValueError("A `data_location` must be provided.")

        self.data_location = str(data_location)
        settings = config["data_set"]["NestedPandasDataset"]
        self.read_parquet_kwargs = settings["read_parquet_kwargs"]

        self.frame = self._load_frame()
        self._register_getters()
        super().__init__(config)

    def _load_frame(self):
        try:
            import nested_pandas as npd
        except ImportError as err:
            raise ImportError(
                "NestedPandasDataset requires the `nested-pandas` package. "
                "Install it with `pip install nested-pandas`."
            ) from err

        return npd.read_parquet(self.data_location, **self.read_parquet_kwargs)

    def _sanitize_name(self, name: str) -> str:
        sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
        if not sanitized:
            return "field"
        if sanitized[0].isdigit():
            return f"field_{sanitized}"
        return sanitized

    def _normalize_value(self, value):
        if isinstance(value, pd.DataFrame):
            if len(value) == 0:
                return {}
            row = value.iloc[0]
            return {col: self._normalize_value(row[col]) for col in row.index}

        if isinstance(value, pd.Series):
            return [self._normalize_value(item) for item in value.tolist()]

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, tuple):
            return [self._normalize_value(item) for item in value]

        return value

    def _register_getters(self) -> None:
        def _make_base_getter(column_name: str):
            def getter(self, idx, _column_name=column_name):
                row = self.frame.iloc[int(idx)]
                return self._normalize_value(row[_column_name])

            return getter

        def _make_nested_getter(nested_name: str):
            def getter(self, idx, _nested_name=nested_name):
                row = self.frame.iloc[int(idx)]
                return self._normalize_value(row[_nested_name])

            return getter

        def _make_nested_subcolumn_getter(nested_name: str, subcolumn_name: str):
            def getter(self, idx, _nested_name=nested_name, _subcolumn_name=subcolumn_name):
                series = self.frame[f"{_nested_name}.{_subcolumn_name}"]
                return self._normalize_value(series.iloc[int(idx)])

            return getter

        def _make_object_id_getter():
            def getter(self, idx):
                return self._normalize_value(self.frame.iloc[int(idx)]["object_id"])

            return getter

        for column_name in self.frame.base_columns:
            method_name = f"get_{self._sanitize_name(column_name)}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_base_getter(column_name), self))

        for nested_name in self.frame.nested_columns:
            method_name = f"get_{self._sanitize_name(nested_name)}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_nested_getter(nested_name), self))

            for subcolumn_name in self.frame.all_columns[nested_name]:
                sub_method_name = f"get_{self._sanitize_name(f'{nested_name}_{subcolumn_name}')}"
                if not hasattr(self, sub_method_name):
                    setattr(
                        self,
                        sub_method_name,
                        MethodType(_make_nested_subcolumn_getter(nested_name, subcolumn_name), self),
                    )

        if self.frame.index.name:
            index_method_name = f"get_{self._sanitize_name(self.frame.index.name)}"
            if not hasattr(self, index_method_name):
                setattr(
                    self,
                    index_method_name,
                    MethodType(lambda self, idx: self._normalize_value(self.frame.index[int(idx)]), self),
                )

        if not hasattr(self, "get_object_id") and "object_id" in self.frame.columns:
            self.get_object_id = MethodType(_make_object_id_getter(), self)

    def __len__(self) -> int:
        return len(self.frame)
