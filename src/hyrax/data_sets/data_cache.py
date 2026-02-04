import logging
import time
from collections.abc import Iterable
from concurrent.futures import Executor
from numbers import Number
from sys import getsizeof
from threading import Thread
from typing import Any

import numpy as np

from hyrax.data_sets.data_provider import DataProvider
from hyrax.tensorboardx_logger import get_tensorboard_logger

logger = logging.getLogger(__name__)
tensorboardx_logger = get_tensorboard_logger()


class DataCache:
    """
    DataCache tracks and manages a caching layer which can be used most effectively if the entirety of a
    training (or inference) epoch fits in system RAM.

    Two configs control this functionality:

    `h.config["data_set"]["use_cache"]` which determines if we are serving data dictionaries out of a cache.
    When set, the first epoch of training fills the cache with tensors, and subsequent epochs are served out
    of the cache.

    `h.config["data_set"]["preload_cache"]` starts a thread which iterates over the dataset/dataloader class
    to completion. The thread pre-loads the cache with tensors independently of the training process. The
    hope is that this thread proceeds faster than the first epoch of training and speeds up the first epoch
    as well.

    In this class we cache the output of DataProvider, before being batched. Users can control the size of
    data cached by only selecting particular fields in their model_inputs specification.

    The class logs to the tensorboard logger in the DataProvider (when configured).
    """

    def __init__(self, config, data_provider: DataProvider):
        """Initialize the DataCache with a Hyrax config.

        Parameters
        ----------
        config : dict
            The Hyrax configuration that defines the data_request.

        data_provider : DataProvider
            The DataProvider object which we are caching for.

        """

        # Grab what we need from data_provider, hang on to self so we can call resolve data func
        self._max_length = len(data_provider)
        self._resolve_data_func = data_provider.resolve_data.__func__
        self._data_provider = data_provider

        # Save config we need
        self._use_cache = config["data_set"]["use_cache"]
        self._preload_cache = config["data_set"]["preload_cache"]

        # Data size and count tracking
        self._data_size_bytes = 0
        self._insert_count = 0

        # On inserts, how many insert operations happen before we log?
        # TODO: This might be a config?
        self.logging_interval = 1000

        # TODO: By design we have a max size at construction time
        # Can we get faster cache access/insert by pre-allocating a
        # vector to hold every possible pointer vs. whatever dict does.
        self._cache_map = {}

        self._preload_thread = None
        if self._preload_cache and self._use_cache:
            self._preload_threads = config["data_set"]["preload_threads"]
            self._preload_thread = Thread(
                name="DataCache-preload-tensor-cache",
                daemon=True,
                target=self._preload_tensor_cache.__func__,  # type: ignore[attr-defined]
                args=(self,),
            )

    def start_preload_thread(self):
        """Start the cache preload thread if configured

        This exists to separate initialization from thread start in DataProvider's
        constructor, so the thread started can always count on a fully initialized DataProvider.
        """
        if self._preload_thread is not None:
            self._preload_thread.start()

    def _idx_check(self, idx):
        if not isinstance(idx, int):
            msg = f"Only integers are allowed as cache keys to DataCache. Got {type(idx)}"
            msg += " as an index type."
            raise RuntimeError(msg)
        if idx >= self._max_length:
            msg = f"Requested index {idx}, but DataCache cache has max length of {self._max_length} "
            msg += "given by the length of the primary dataset in model_inputs."
            raise IndexError(msg)

    def try_fetch(self, idx: int) -> dict | None:
        """
        Try to fetch a data_dict from the cache.

        Parameters
        ----------
        idx : int
            The DataProvider index of the data dict

        Returns
        -------
        Optional[dict]
            The data dict from the cache, None on a cache miss.
        """
        if self._use_cache:
            self._idx_check(idx)
            return self._cache_map.get(idx, None)
        return None

    def insert_into_cache(self, idx: int, data: dict[str, dict[str, Any]]):
        """Insert a data dict into the cache

        Parameters
        ----------
        idx : int
            Index of the data dict
        data : dict[str, dict[str, Any]]
            The data dict
        """
        start_time = time.monotonic_ns()
        prefix = self.__class__.__name__
        if self._use_cache:
            self._idx_check(idx)
            self._insert_count += 1
            old_value = self._cache_map.get(idx, None)
            if old_value is not None:
                self._data_size_bytes -= DataCache._data_size(old_value)

            self._cache_map[idx] = data
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
                # We don't recurse here because .nbytes and getsizeof() work across numpy and torch
                # and we want to keep torch objects *out* of the cache, but we don't mind numpy objects
                # who's memory is actually owned by torch.
                seen.add(id(data.base))
                total_data_size += data.base.nbytes + getsizeof(data.base)
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

    def _preload_tensor_cache(self):
        """
        Preload all tensors in the dataset using multiple threads.
        """
        from concurrent.futures import ThreadPoolExecutor

        logger.info("Preloading Data cache...")
        prefix = self.__class__.__name__

        with ThreadPoolExecutor(max_workers=self._preload_threads) as executor:
            fetched_data = self._lazy_map_executor(executor, range(self._max_length))

            start_time = time.monotonic_ns()
            for idx, data_item in enumerate(fetched_data):
                self.insert_into_cache(idx, data_item)

                # Output timing every 1k tensors
                if idx % 1_000 == 0 and idx != 0:
                    tensorboardx_logger.log_duration_ts(f"{prefix}/preload_1k_obj_s", start_time)
                    start_time = time.monotonic_ns()

    def _lazy_map_executor(self, executor: Executor, idxs: Iterable[int]):
        """
        Lazy evaluation version of concurrent.futures.Executor.map().

        This limits memory usage during preloading by keeping only a small
        number of data dictionaries in memory at once.

        Parameters
        ----------
        executor : concurrent.futures.Executor
            An executor for running futures
        idxs : Iterable[int]
            An iterable list of DataProvider indexes

        Yields
        ------
        Iterator[torch.Tensor]
            An iterator over torch tensors, lazily loaded
        """
        from concurrent.futures import FIRST_COMPLETED, Future, wait

        ## We use self._preload_threads here as both the number of workers and
        ## the number of in-flight futures that this lazy map executor keeps running
        ##
        ## The goal here is actually maximum filesystem performance on a high-latency filesystem
        ## Currently the defaults are tuned for UW's HYAK Klone filesystem, where 50 threads is optimal.
        ##
        ## A better implementation would look at how long the main
        ## thread of the preloader spends waiting for workers. For a balanced situation where
        ## there are the right number of workers to fully exercise the I/O system:
        ##
        ##  N = number of wokers
        ##  t_w = Wall clock time a worker/future takes to execute (averaged over some period, because I/O
        ##        is bursty.)
        ##  t_p = Wall clock time the preload thread waits between workers completing
        ##
        ##  t_p == t_w/N
        ##
        ## If the preload thread is waiting too long t_p > t_w/N -> Increase the number of futures in flight,
        ##   spawn more workers as needed so every in-flight future can have a worker
        ##
        ## If we're within some epsilon of t_p == t_w/N then keep workers constant
        ##
        ## If t_p < t_w/N then reduce the number of futures in flight
        ##
        ## We would need to figure out a reasonable averaging/adjustment time that's long enough to capture
        ## a stable time-average on most systems while not giving away speed
        ##
        ## We also might want to put all of this in a log basis i.e. log(t_p) + epsilon == log(t_w) - log(N)
        ## This would mean the epsilon thresholding would give the algorithm a larger target at larger
        ## thread counts, which matches how most I/O systems behave. Linearizing the problem in this way
        ## would also reduce instances where the algorithm changing the number of threads throws it into
        ## a feedback loop.
        ##
        ## Sadly some of the logic for this would be inside ThreadPoolExecutor, becuase we need to post-hoc
        ## alter the number of worker threads after creating the object. The alternative would be
        ## preallocating worker threads that we don't use.
        max_futures = self._preload_threads

        queue: list[Future[dict]] = []
        in_progress: set[Future[dict]] = set()
        idx_iterator = iter(idxs)

        try:
            while True:
                for _ in range(max_futures - len(in_progress)):
                    idx = next(idx_iterator)
                    future = executor.submit(self._resolve_data_func, self._data_provider, idx)  # type: ignore[attr-defined]
                    queue.append(future)
                    in_progress.add(future)

                _, in_progress = wait(in_progress, return_when=FIRST_COMPLETED)

                while queue and queue[0].done():
                    yield queue.pop(0).result()

        except StopIteration:
            wait(queue)
            for future in queue:
                try:
                    result = future.result()
                except Exception as e:
                    raise e
                else:
                    yield result
