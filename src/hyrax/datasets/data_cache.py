import logging
import time
from numbers import Number
from sys import getsizeof
from typing import Any

import numpy as np

from hyrax.tensorboardx_logger import get_tensorboard_logger

logger = logging.getLogger(__name__)
tensorboardx_logger = get_tensorboard_logger()


class DataCache:
    """Per-dataset caching layer.

    Each ``DataCache`` instance belongs to a single dataset and manages two
    cache maps:

    * **base cache** — keyed by ``idx`` (an int), stores the result of
      ``get_<field>`` calls.
    * **augment cache** — keyed by a caller-provided key (typically from
      ``augment_cache_key``), stores augmented results.

    One config controls this functionality:

    ``h.config["data_set"]["use_cache"]`` — when True, data dicts are cached
    after the first access so subsequent accesses are served from memory.
    """

    def __init__(self, use_cache: bool):
        """Initialize the DataCache.

        Parameters
        ----------
        use_cache : bool
            Whether caching is enabled.
        """
        self._use_cache = use_cache

        self._data_size_bytes = 0
        self._insert_count = 0
        self.logging_interval = 1000

        self._base_cache: dict[int, dict] = {}
        self._augment_cache: dict[Any, dict] = {}

    def try_fetch_base(self, idx: int) -> dict | None:
        """Try to fetch base (non-augmented) cached data.

        Parameters
        ----------
        idx : int
            The dataset-local index.

        Returns
        -------
        dict | None
            The cached field dict, or ``None`` on miss.
        """
        if not self._use_cache:
            return None
        return self._base_cache.get(idx)

    def try_fetch_augmented(self, key) -> dict | None:
        """Try to fetch augmented cached data.

        Parameters
        ----------
        key
            The augment cache key (from ``augment_cache_key``).

        Returns
        -------
        dict | None
            The cached augmented field dict, or ``None`` on miss.
        """
        if not self._use_cache:
            return None
        return self._augment_cache.get(key)

    def insert_base(self, idx: int, data: dict[str, Any]):
        """Insert base (non-augmented) field data into the cache.

        Parameters
        ----------
        idx : int
            The dataset-local index.
        data : dict[str, Any]
            The field data dict to cache.
        """
        if not self._use_cache:
            return
        self._do_insert(self._base_cache, idx, data)

    def insert_augmented(self, key, data: dict[str, Any]):
        """Insert augmented field data into the cache.

        Parameters
        ----------
        key
            The augment cache key (from ``augment_cache_key``).
        data : dict[str, Any]
            The augmented field data dict to cache.
        """
        if not self._use_cache:
            return
        self._do_insert(self._augment_cache, key, data)

    def _do_insert(self, cache_map: dict, cache_key, data: dict[str, Any]):
        start_time = time.monotonic_ns()
        prefix = self.__class__.__name__

        self._insert_count += 1
        old_value = cache_map.get(cache_key)
        if old_value is not None:
            self._data_size_bytes -= DataCache._data_size(old_value)

        cache_map[cache_key] = data
        self._data_size_bytes += DataCache._data_size(data)
        tensorboardx_logger.log_duration_ts(f"{prefix}/cache_insert_s", start_time)
        if self._insert_count % self.logging_interval == 0 and self._insert_count != 0:
            tensorboardx_logger.log_scalar_ts(f"{prefix}/cache_count", self._insert_count)
            tensorboardx_logger.log_scalar_ts(f"{prefix}/cache_bytes", self._data_size_bytes)

    @staticmethod
    def _data_size(data, seen: set[int] | None = None) -> int:
        if seen is None:
            seen = set([])

        # Handle objects we've seen before
        if id(data) in seen:
            return 0
        else:
            seen.add(id(data))

        total_data_size = 0
        # Handle different types
        # For dicts we need to count the keys and val separate from the dict,
        # because they don't own their data.
        if isinstance(data, dict):
            total_data_size += getsizeof(data)
            for k, v in data.items():
                total_data_size += DataCache._data_size(k, seen)
                total_data_size += DataCache._data_size(v, seen)
        # List reported size also does not include the size of the list items
        elif isinstance(data, list):
            total_data_size += getsizeof(data)
            for i in data:
                total_data_size += DataCache._data_size(i, seen)
        # Numpy arrays report a size that is their entire size including their members when they
        # own the data (data.base = None).
        # When they don't own the data (data.base= <some object>) only overhead is reported
        elif isinstance(data, np.ndarray):
            if data.base is None:
                # Owns its data - count the actual data
                total_data_size += data.nbytes + getsizeof(data)
            elif id(data.base) not in seen:
                # We haven't seen the base object. Add it to seen, and assume for the view we're
                # examining now that the whole base object is necessary
                #
                # We don't recurse here because .nbytes and getsizeof() work the same way
                # across numpy and torch. While we want to keep torch objects *out* of the cache,
                # we don't mind numpy objects who's memory is actually owned by torch due to how they
                # were constructed.
                seen.add(id(data.base))

                # Sometimes a numpy object is created from a not numpy/torch shaped thing so data.base
                # doesn't follow the numpy/torch convention of getsizeof(obj) = bookeeping overhead
                # real memory size elsewhere.
                #
                # Fall back to assuming to only getsize(obj) is the whole picture of the base object when
                # this happens.
                #
                # For example: a numpy object created from a PIL Image has a bytestring as data.base which
                # comes from the PIL Image's .tobytes() method. In this case getsizeof(data.base) works fine
                # on its own to get the size contribution of the base object.
                base_nbytes = data.base.nbytes if hasattr(data.base, "nbytes") else 0

                total_data_size += base_nbytes + getsizeof(data.base)
            else:
                # Is a view - with a base we've seen before, just add overhead
                total_data_size += getsizeof(data)
        # Basic data types are just their own size
        elif isinstance(data, (np.number, Number, type(None), np.bool)):
            total_data_size += getsizeof(data)
        # String types are also just their own size
        elif isinstance(data, (np.character, str)):
            total_data_size += getsizeof(data)
        # Catch all types we haven't written handling for.
        else:
            msg = f"Unsupported type for cache: {type(data)} \n"
            msg += "Please only use python basic data types and numpy types."
            raise RuntimeError(msg)

        return total_data_size
