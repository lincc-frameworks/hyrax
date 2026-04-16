from __future__ import annotations

import bisect
import logging
from collections import OrderedDict
from pathlib import Path
from types import MethodType
from typing import Any

from hyrax.datasets.dataset_registry import HyraxDataset

logger = logging.getLogger(__name__)


class HATSPartitionIndex:
    """Partition row-count index used to resolve global row indices lazily."""

    def __init__(self, catalog: Any, strict_metadata: bool = True):
        self.partition_row_counts = self._build_partition_row_counts(catalog, strict_metadata=strict_metadata)
        self._prefix_counts = []

        running = 0
        for count in self.partition_row_counts:
            running += count
            self._prefix_counts.append(running)

    @property
    def total_rows(self) -> int:
        return 0 if len(self._prefix_counts) == 0 else self._prefix_counts[-1]

    def resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(f"Index {idx} is out of range for dataset of length {self.total_rows}.")

        partition_id = bisect.bisect_right(self._prefix_counts, idx)
        previous_total = 0 if partition_id == 0 else self._prefix_counts[partition_id - 1]
        local_offset = idx - previous_total
        return partition_id, local_offset

    @staticmethod
    def _build_partition_row_counts(catalog: Any, strict_metadata: bool = True) -> list[int]:
        ddf = getattr(catalog, "_ddf", None)
        if ddf is None:
            ddf = getattr(catalog, "ddf", None)

        if ddf is not None and hasattr(ddf, "map_partitions"):
            try:
                counts = ddf.map_partitions(len).compute()
                return [int(x) for x in counts]
            except Exception:
                if strict_metadata:
                    msg = "Could not derive per-partition row counts from catalog metadata."
                    raise ValueError(msg) from None
                logger.debug("Falling back to eager row-count compute for HATS catalog.", exc_info=True)

        try:
            fallback_length = len(catalog.compute())
            logger.debug("Using eager fallback row-count metadata for HATS catalog.")
            return [fallback_length]
        except Exception as exc:
            msg = "Could not derive HATS partition row counts; unable to build lazy row index."
            raise ValueError(msg) from exc


class HATSRowBundleCache:
    """Simple LRU cache for partition row bundles."""

    def __init__(self, max_bundles: int = 64):
        self.max_bundles = max(1, int(max_bundles))
        self._cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[Any, ...]):
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            logger.debug("HATS row-bundle cache hit for key=%s", key)
            return self._cache[key]

        self.misses += 1
        logger.debug("HATS row-bundle cache miss for key=%s", key)
        return None

    def set(self, key: tuple[Any, ...], value: Any):
        self._cache[key] = value
        self._cache.move_to_end(key)

        if len(self._cache) > self.max_bundles:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("HATS row-bundle cache evicted key=%s", evicted_key)


class HATSLazyAccessor:
    """Lazy row fetch/access helper for Hyrax HATS datasets."""

    def __init__(
        self,
        catalog: Any,
        partition_index: HATSPartitionIndex,
        cache: HATSRowBundleCache,
        *,
        bundle_size: int = 256,
        project_columns: str | list[str] = "auto",
        required_columns: list[str] | None = None,
    ):
        self.catalog = catalog
        self.partition_index = partition_index
        self.cache = cache
        self.bundle_size = max(1, int(bundle_size))
        self.project_columns = project_columns
        self.required_columns = required_columns or []

    def _bundle_key(self, partition_id: int, bundle_start: int, columns: list[str]) -> tuple[Any, ...]:
        return (partition_id, bundle_start, self.bundle_size, tuple(columns))

    def _projected_columns(self, requested_column: str | None) -> list[str]:
        all_columns = self._catalog_columns()
        if self.project_columns == "all":
            return list(all_columns)

        requested = [requested_column] if requested_column else []

        if isinstance(self.project_columns, list):
            columns = list(dict.fromkeys(self.project_columns + requested))
            return columns

        columns = list(dict.fromkeys(self.required_columns + requested))
        return columns if len(columns) > 0 else requested

    def _catalog_columns(self) -> list[str]:
        columns = getattr(self.catalog, "columns", None)
        if columns is not None:
            return list(columns)

        ddf = getattr(self.catalog, "_ddf", None) or getattr(self.catalog, "ddf", None)
        if ddf is not None and hasattr(ddf, "columns"):
            return list(ddf.columns)

        return []

    def _read_partition_dataframe(self, partition_id: int, columns: list[str]):
        ddf = getattr(self.catalog, "_ddf", None) or getattr(self.catalog, "ddf", None)
        if ddf is not None and hasattr(ddf, "get_partition"):
            partition = ddf.get_partition(partition_id)
            if len(columns) > 0:
                partition = partition[columns]
            return partition.compute()

        # Conservative fallback path: compute full catalog and treat as one partition.
        df = self.catalog.compute()
        if len(columns) > 0:
            return df[columns]
        return df

    def get_row(self, idx: int):
        partition_id, local_offset = self.partition_index.resolve(idx)
        columns = self._projected_columns(requested_column=None)

        bundle_start = (local_offset // self.bundle_size) * self.bundle_size
        cache_key = self._bundle_key(partition_id, bundle_start, columns)
        bundle = self.cache.get(cache_key)

        if bundle is None:
            partition_df = self._read_partition_dataframe(partition_id, columns)
            bundle = partition_df.iloc[bundle_start : bundle_start + self.bundle_size]
            self.cache.set(cache_key, bundle)

        return bundle.iloc[local_offset - bundle_start]

    def get_value(self, idx: int, column: str):
        partition_id, local_offset = self.partition_index.resolve(idx)
        columns = self._projected_columns(requested_column=column)

        bundle_start = (local_offset // self.bundle_size) * self.bundle_size
        cache_key = self._bundle_key(partition_id, bundle_start, columns)
        bundle = self.cache.get(cache_key)

        if bundle is None:
            partition_df = self._read_partition_dataframe(partition_id, columns)
            bundle = partition_df.iloc[bundle_start : bundle_start + self.bundle_size]
            self.cache.set(cache_key, bundle)

        import numpy as np
        import pandas as pd

        ret_val = bundle.iloc[local_offset - bundle_start][column]
        if isinstance(ret_val, pd.Series):
            ret_val = ret_val.to_numpy()
        elif isinstance(ret_val, (list, tuple)):
            ret_val = np.asarray(ret_val)
        return ret_val


class HyraxHATSDataset(HyraxDataset):
    """Generic Hyrax dataset for HATS catalogs loaded through LSDB.

    Phase-2 default behavior is lazy and partition-aware, with optional fallback
    to eager compute if metadata access is unavailable.
    """

    def __init__(self, config: dict, data_location: Path = None):
        if data_location is None:
            raise ValueError("A `data_location` to a HATS catalog must be provided.")

        self.data_location = data_location
        self._config = config

        hats_config = self._extract_hats_config(config)

        self.catalog = self._open_hats_catalog(data_location)
        self.column_names = self._catalog_columns()

        strict_metadata = hats_config.get("strict_metadata", True)
        bundle_size = hats_config.get("bundle_size", 256)
        max_cached_bundles = hats_config.get("max_cached_bundles", 64)
        project_columns = hats_config.get("project_columns", "auto")

        required_columns = self._required_columns(config, self.column_names)

        partition_index = HATSPartitionIndex(self.catalog, strict_metadata=strict_metadata)
        cache = HATSRowBundleCache(max_bundles=max_cached_bundles)
        self._accessor = HATSLazyAccessor(
            self.catalog,
            partition_index,
            cache,
            bundle_size=bundle_size,
            project_columns=project_columns,
            required_columns=required_columns,
        )

        def _make_getter(column: str):
            def getter(self, idx: int, _col: str = column):
                return self._accessor.get_value(idx, _col)

            return getter

        for col in self.column_names:
            method_name = f"get_{col}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(col), self))

        super().__init__(config)

    @staticmethod
    def _extract_hats_config(config: dict) -> dict:
        request = config.get("data_request", {})
        for split_cfg in request.values():
            if not isinstance(split_cfg, dict):
                continue
            for dataset_cfg in split_cfg.values():
                if isinstance(dataset_cfg, dict) and dataset_cfg.get("dataset_class") == "HyraxHATSDataset":
                    return dataset_cfg.get("hats", {})
        return {}

    @staticmethod
    def _required_columns(config: dict, all_columns: list[str]) -> list[str]:
        request = config.get("data_request", {})
        required = []

        for split_cfg in request.values():
            if not isinstance(split_cfg, dict):
                continue
            for dataset_cfg in split_cfg.values():
                if not isinstance(dataset_cfg, dict):
                    continue
                fields = dataset_cfg.get("fields", [])
                if isinstance(fields, list):
                    required.extend(fields)
                primary_id_field = dataset_cfg.get("primary_id_field")
                if primary_id_field:
                    required.append(primary_id_field)

        required = [col for col in dict.fromkeys(required) if col in all_columns]
        return required

    @staticmethod
    def _open_hats_catalog(data_location: Path):
        import lsdb

        if hasattr(lsdb, "open_catalog"):
            try:
                return lsdb.open_catalog(data_location)
            except Exception:
                logger.debug("Failed to open HATS catalog via lsdb.open_catalog; falling back.", exc_info=True)

        if hasattr(lsdb, "read_hats"):
            return lsdb.read_hats(data_location)

        raise RuntimeError("Installed LSDB does not expose open_catalog or read_hats.")

    def _catalog_columns(self) -> list[str]:
        columns = getattr(self.catalog, "columns", None)
        if columns is not None:
            return list(columns)

        ddf = getattr(self.catalog, "_ddf", None) or getattr(self.catalog, "ddf", None)
        if ddf is not None and hasattr(ddf, "columns"):
            return list(ddf.columns)

        # Fallback for unusual catalog wrappers.
        return list(self.catalog.compute().columns)

    def __len__(self) -> int:
        return self._accessor.partition_index.total_rows

    def __getitem__(self, idx):
        """Currently required by Hyrax machinery, but likely to be phased out."""
        row = self._accessor.get_row(idx)
        return {"data": {col: row[col] for col in self.column_names}}

    def sample_data(self):
        """Return the first record in dictionary form as a sample."""
        if len(self) == 0:
            return {"data": {col: None for col in self.column_names}}
        row = self._accessor.get_row(0)
        return {"data": {col: row[col] for col in self.column_names}}
