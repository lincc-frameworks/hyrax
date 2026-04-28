from pathlib import Path
from types import MethodType

from hyrax.datasets.dataset_registry import HyraxDataset


class NestedPandasDataset(HyraxDataset):
    """Hyrax dataset wrapping a nested-pandas parquet file."""

    def __init__(self, config: dict, data_location: Path | str | None = None):
        if data_location is None:
            raise ValueError("A `data_location` must be provided.")

        try:
            import nested_pandas as npd
        except ImportError as err:
            raise ImportError(
                "NestedPandasDataset requires the `nested-pandas` package. "
                "Install it with `pip install nested-pandas`."
            ) from err

        settings = config["data_set"]["NestedPandasDataset"]
        self.data_location = str(data_location)
        self.read_parquet_kwargs = settings["read_parquet_kwargs"]

        self.frame = npd.read_parquet(self.data_location, **self.read_parquet_kwargs)
        self._register_getters()
        super().__init__(config)

    def _register_getters(self) -> None:
        def _make_getter(col):
            def getter(self, idx, _col=col):
                return self.frame.iloc[int(idx)][_col]

            return getter

        for col in self.frame.columns:
            method_name = f"get_{col}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(col), self))

    def __len__(self) -> int:
        return len(self.frame)
