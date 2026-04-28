import re
from pathlib import Path
from types import MethodType

import pandas as pd

from hyrax.datasets.dataset_registry import HyraxDataset


def _normalize_field_name(field_name: str) -> str:
    """Normalize a source column name into a valid Hyrax field name."""
    return re.sub(r"\W", "_", field_name)


class NestedPandasDataset(HyraxDataset):
    """A Hyrax dataset that reads nested-pandas parquet files."""

    def __init__(self, config: dict, data_location: Path | str | None = None):
        if data_location is None:
            raise ValueError("A `data_location` must be provided.")

        try:
            import nested_pandas
        except ImportError as err:
            raise ImportError(
                "NestedPandasDataset requires the `nested-pandas` package. "
                "Install it with `pip install nested-pandas`."
            ) from err

        settings = config["data_set"]["NestedPandasDataset"]
        self.data_location = str(data_location)
        self.read_parquet_kwargs = settings["read_parquet_kwargs"]
        self.nested_frame = nested_pandas.read_parquet(self.data_location, **self.read_parquet_kwargs)
        self.field_map = self._build_field_map()
        self._register_getters()

        super().__init__(config)

    def _build_field_map(self) -> dict[str, tuple[str, str] | tuple[str, str, str]]:
        field_map: dict[str, tuple[str, str] | tuple[str, str, str]] = {}

        for field_name in self.nested_frame.columns:
            normalized_name = _normalize_field_name(field_name)
            self._register_field_mapping(field_map, normalized_name, ("base", field_name))

        for nested_column in self.nested_frame.nested_columns:
            for subcolumn in self.nested_frame.all_columns[nested_column]:
                normalized_name = (
                    f"{_normalize_field_name(nested_column)}__{_normalize_field_name(subcolumn)}"
                )
                self._register_field_mapping(
                    field_map,
                    normalized_name,
                    ("nested", nested_column, subcolumn),
                )

        return field_map

    def _register_field_mapping(
        self,
        field_map: dict[str, tuple[str, str] | tuple[str, str, str]],
        normalized_name: str,
        source: tuple[str, str] | tuple[str, str, str],
    ) -> None:
        existing_source = field_map.get(normalized_name)
        if existing_source is not None and existing_source != source:
            raise ValueError(
                "NestedPandasDataset found multiple source columns that normalize to "
                f"`{normalized_name}`. Rename one of the columns to avoid a getter collision."
            )
        field_map[normalized_name] = source

    def _register_getters(self) -> None:
        def _make_base_getter(field_name: str):
            def getter(self, idx, _field_name=field_name):
                return self.nested_frame.iloc[int(idx)][_field_name]

            return getter

        def _make_nested_getter(nested_column: str, subcolumn: str):
            def getter(self, idx, _nested_column=nested_column, _subcolumn=subcolumn):
                nested_table = self.nested_frame.iloc[int(idx)][_nested_column]
                values = nested_table[_subcolumn]
                if isinstance(values, pd.Series):
                    return values.to_numpy()
                return values

            return getter

        for field_name, source in self.field_map.items():
            method_name = f"get_{field_name}"
            if hasattr(self, method_name):
                continue

            if source[0] == "base":
                setattr(self, method_name, MethodType(_make_base_getter(source[1]), self))
            else:
                setattr(self, method_name, MethodType(_make_nested_getter(source[1], source[2]), self))

    def __len__(self) -> int:
        return len(self.nested_frame)
