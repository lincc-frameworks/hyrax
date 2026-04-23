from pathlib import Path
from types import MethodType

from hyrax.datasets.dataset_registry import HyraxDataset


class LanceDBDataset(HyraxDataset):
    """A minimal Hyrax wrapper around a LanceDB table."""

    def __init__(self, config: dict, data_location: Path | str | None = None):
        if data_location is None:
            raise ValueError("A `data_location` must be provided.")

        try:
            import lancedb
        except ImportError as err:
            raise ImportError(
                "LanceDBDataset requires the `lancedb` package. "
                "Install it with `pip install lancedb`."
            ) from err

        settings = config["data_set"]["LanceDBDataset"]
        self.data_location = str(data_location)
        self.table_name = settings["table_name"]
        self.connect_kwargs = settings["connect_kwargs"]
        self.open_table_kwargs = settings["open_table_kwargs"]

        self.db = lancedb.connect(self.data_location, **self.connect_kwargs)
        self.table = self.db.open_table(self.table_name, **self.open_table_kwargs)
        self.lance_dataset = self.table.to_lance()

        self._register_getters()
        super().__init__(config)

    def _all_available_fields(self) -> list[str]:
        return list(self.table.schema.names)

    def _register_getters(self) -> None:
        def _make_getter(field_name: str):
            def getter(self, idx, _field_name=field_name):
                row = self.lance_dataset.take([int(idx)])
                return row[_field_name][0].as_py()

            return getter

        for field_name in self._all_available_fields():
            method_name = f"get_{field_name}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(field_name), self))

    def __len__(self) -> int:
        return self.table.count_rows()
