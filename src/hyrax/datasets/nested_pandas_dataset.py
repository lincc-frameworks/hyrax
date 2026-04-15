from pathlib import Path
from types import MethodType

from hyrax.datasets.dataset_registry import HyraxDataset


class NestedPandasDataset(HyraxDataset):
    """A minimal Hyrax wrapper around ``nested_pandas.read_parquet``."""

    def __init__(self, config: dict, data_location: Path | str | None = None):
        if data_location is None:
            raise ValueError("A `data_location` must be provided.")

        self.data_location = str(data_location)
        settings = config["data_set"]["NestedPandasDataset"]
        self.read_kwargs = settings["read_kwargs"]

        self.nested_frame = self._load_nested_frame(self.read_kwargs)

        self._register_getters()
        super().__init__(config)

    def _load_nested_frame(self, read_kwargs: dict):
        try:
            import nested_pandas as npd
        except ImportError as err:
            raise ImportError(
                "NestedPandasDataset requires the `nested-pandas` package. "
                "Install it with `pip install nested-pandas`."
            ) from err

        return npd.read_parquet(self.data_location, **read_kwargs)

    def _all_available_fields(self) -> list[str]:
        fields = list(self.nested_frame.columns)
        if hasattr(self.nested_frame, "get_subcolumns"):
            fields.extend(self.nested_frame.get_subcolumns())
        return fields

    def _register_getters(self) -> None:
        def _make_getter(field_name: str):
            def getter(self, idx, _field_name=field_name):
                import pandas as pd

                retval = self.nested_frame[_field_name].loc[self.nested_frame.index[idx]]
                return retval.to_numpy() if isinstance(retval, (pd.DataFrame, pd.Series)) else retval

            return getter

        for field_name in self._all_available_fields():
            method_name = f"get_{field_name}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(field_name), self))

    def __len__(self) -> int:
        return len(self.nested_frame)
