from collections import OrderedDict
from pathlib import Path
from types import MethodType

from hyrax.datasets.dataset_registry import HyraxDataset

_ROW_CACHE_SIZE = 16


class LanceDBDataset(HyraxDataset):
    """A minimal Hyrax wrapper around a LanceDB table."""

    def __init__(self, config: dict, data_location: Path | str | None = None):
        if data_location is None:
            raise ValueError("A `data_location` must be provided.")

        try:
            import lancedb
        except ImportError as err:
            raise ImportError(
                "LanceDBDataset requires the `lancedb` package. Install it with `pip install lancedb`."
            ) from err

        settings = config["data_set"]["LanceDBDataset"]
        self.data_location = str(data_location)
        self.table_name = settings["table_name"]
        self.connect_kwargs = settings["connect_kwargs"]
        self.open_table_kwargs = settings["open_table_kwargs"]

        self.db = lancedb.connect(self.data_location, **self.connect_kwargs)
        self.table_name = self._resolve_table_name(self.table_name)
        self.table = self.db.open_table(self.table_name, **self.open_table_kwargs)
        self.lance_dataset = self.table.to_lance()
        self._row_cache: OrderedDict = OrderedDict()

        self._register_getters()
        super().__init__(config)

    def _all_available_fields(self) -> list[str]:
        return list(self.table.schema.names)

    def _get_row(self, idx: int):
        """Return the PyArrow record-batch for *idx*, using a small FIFO row cache.

        Caching avoids redundant ``lance_dataset.take`` calls when multiple
        ``get_<field>`` accessors are invoked for the same sample index, which is
        the common pattern when DataProvider resolves all fields for a single item.
        The cache holds at most ``_ROW_CACHE_SIZE`` rows; the oldest entry is
        evicted once that limit is reached.
        """
        if idx not in self._row_cache:
            if len(self._row_cache) >= _ROW_CACHE_SIZE:
                self._row_cache.popitem(last=False)
            self._row_cache[idx] = self.lance_dataset.take([idx])
        return self._row_cache[idx]

    def _resolve_table_name(self, configured_table_name) -> str:
        if isinstance(configured_table_name, str) and configured_table_name:
            return configured_table_name

        table_names = self.db.table_names()
        if len(table_names) == 1:
            return table_names[0]

        available_tables = ", ".join(table_names) if len(table_names) > 0 else "(none)"
        raise RuntimeError(
            "LanceDBDataset could not infer a table to open because `table_name` is unset "
            "and the database does not have exactly one table. "
            "Set `config['data_set']['LanceDBDataset']['table_name']` "
            f"to one of: {available_tables}"
        )

    def _register_getters(self) -> None:
        def _make_getter(field_name: str):
            def getter(self, idx, _field_name=field_name):
                row = self._get_row(int(idx))
                return row[_field_name][0].as_py()

            return getter

        for field_name in self._all_available_fields():
            method_name = f"get_{field_name}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(field_name), self))

    def __len__(self) -> int:
        return self.table.count_rows()
