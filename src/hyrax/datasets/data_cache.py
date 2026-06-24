import logging
import time
from numbers import Number
from sys import getsizeof
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hyrax.datasets.dataset_registry import HyraxDataset

import numpy as np

from hyrax.tensorboardx_logger import get_tensorboard_logger

logger = logging.getLogger(__name__)
tensorboardx_logger = get_tensorboard_logger()


class DataCache:
    """Per-dataset caching layer for DataProvider.

    Each dataset (friendly name) gets two cache maps:

    * **base cache** — keyed by ``real_idx`` (an int), stores the result of
      ``get_<field>`` calls.  No dataset method is called to produce the key.
    * **augment cache** — keyed by the return value of the dataset's
      ``augment_cache_key`` method, stores augmented results.  Only populated
      when the dataset opts in by returning a non-None key.

    ``try_fetch`` checks the augment cache first (when applicable), then falls
    back to the base cache.

    One config controls this functionality:

    ``h.config["data_set"]["use_cache"]`` — when True, data dicts are cached
    after the first access so subsequent accesses are served from memory.
    """

    def __init__(
        self,
        config: dict,
        datasets: dict[str, "HyraxDataset"],
        augment_active: dict[str, bool],
    ):
        """Initialize the DataCache.

        Parameters
        ----------
        config : dict
            The Hyrax configuration.
        datasets : dict[str, HyraxDataset]
            Mapping of friendly_name to dataset instance. Used to call
            ``augment_cache_key`` for augmented data caching.
        augment_active : dict[str, bool]
            Mapping of friendly_name to whether augmentation is active
            for that dataset. When True, ``try_fetch`` will check the
            augment cache before falling back to the base cache.
        """
        self._use_cache = config["data_set"]["use_cache"]
        self._datasets = datasets
        self._augment_active = augment_active

        self._data_size_bytes = 0
        self._insert_count = 0
        self.logging_interval = 1000

        self._base_cache: dict[str, dict[int, dict]] = {name: {} for name in datasets}
        self._augment_cache: dict[str, dict[np.int64, dict]] = {name: {} for name in datasets}

    def try_fetch(
        self,
        friendly_name: str,
        real_idx: int,
        rng_seed: np.int64 | None = None,
    ) -> tuple[dict | None, bool]:
        """Try to fetch cached data for a single dataset.

        When augmentation is active and ``rng_seed`` is provided, this checks
        the augment cache first.  On miss it falls back to the base cache.

        Parameters
        ----------
        friendly_name : str
            The dataset friendly name.
        real_idx : int
            The dataset-local index.
        rng_seed : np.int64 | None
            The augmentation RNG seed, or None for non-augmented access.

        Returns
        -------
        tuple[dict | None, bool]
            ``(data, already_augmented)`` where ``data`` is the cached
            field dict or ``None`` on miss, and ``already_augmented``
            indicates whether the cached data includes augmentation.
        """
        if not self._use_cache:
            return None, False

        # When augmentation is active, try augment cache first
        if self._augment_active.get(friendly_name, False) and rng_seed is not None:
            aug_key = self._datasets[friendly_name].augment_cache_key(real_idx, rng_seed)
            if aug_key is not None:
                cached = self._augment_cache[friendly_name].get(aug_key)
                if cached is not None:
                    return cached, True

        # Try base cache — keyed directly by index, no method call
        cached = self._base_cache[friendly_name].get(real_idx)
        if cached is not None:
            return cached, False

        return None, False

    def insert_base(
        self,
        friendly_name: str,
        real_idx: int,
        data: dict[str, Any],
    ):
        """Insert base (non-augmented) field data into the cache.

        Parameters
        ----------
        friendly_name : str
            The dataset friendly name.
        real_idx : int
            The dataset-local index (used directly as cache key).
        data : dict[str, Any]
            The field data dict to cache.
        """
        if not self._use_cache:
            return
        self._do_insert(self._base_cache[friendly_name], real_idx, data)

    def insert_augmented(
        self,
        friendly_name: str,
        real_idx: int,
        rng_seed: np.int64,
        data: dict[str, Any],
    ):
        """Insert augmented field data into the cache.

        Calls ``augment_cache_key`` to determine the cache key. If the key
        is ``None``, this is a no-op (the dataset opted out of caching
        augmented data).

        Parameters
        ----------
        friendly_name : str
            The dataset friendly name.
        real_idx : int
            The dataset-local index.
        rng_seed : np.int64
            The augmentation RNG seed.
        data : dict[str, Any]
            The augmented field data dict to cache.
        """
        if not self._use_cache:
            return
        cache_key = self._datasets[friendly_name].augment_cache_key(real_idx, rng_seed)
        if cache_key is None:
            return
        self._do_insert(self._augment_cache[friendly_name], cache_key, data)

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
