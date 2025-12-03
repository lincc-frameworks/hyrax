import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from concurrent.futures import Executor
from threading import Thread

logger = logging.getLogger(__name__)


class TensorCacheMixin(ABC):
    """
    Mixin class providing in-memory tensor caching functionality for datasets.

    This mixin provides:
    - use_cache: Cache tensors in memory after first load
    - preload_cache: Preload all tensors in background thread
    - Efficient tensor cache management with hit/miss tracking
    - Background preloading with parallel processing

    Classes using this mixin must implement:
    - _load_tensor_for_cache(object_id: str) -> torch.Tensor
    - ids() -> Generator[str] (iterator over object IDs)
    - __len__() -> int
    """

    def _init_tensor_cache(self, config):
        """Initialize tensor caching. Call this from __init__ after other setup."""
        self.use_cache = config["data_set"]["use_cache"]

        # Initialize cache storage and timing
        from torch import Tensor

        self.tensors: dict[str, Tensor] = {}
        self.tensorboard_start_ns = time.monotonic_ns()
        self.tensorboardx_logger = None

        # Start preload thread if configured
        if config["data_set"]["preload_cache"] and self.use_cache:
            self.preload_thread = Thread(
                name=f"{self.__class__.__name__}-preload-tensor-cache",
                daemon=True,
                target=self._preload_tensor_cache.__func__,  # type: ignore[attr-defined]
                args=(self,),
            )
            self.preload_thread.start()

    @abstractmethod
    def _load_tensor_for_cache(self, object_id: str):
        """
        Load tensor for the given object_id. Must be implemented by subclasses.

        Parameters
        ----------
        object_id : str
            The object ID to load tensor for

        Returns
        -------
        torch.Tensor
            The loaded tensor
        """
        pass

    @abstractmethod
    def ids(self, log_every: int | None = None) -> Generator[str, None, None]:
        """
        Iterator over all object IDs. Must be implemented by subclasses.

        Parameters
        ----------
        log_every : Optional[int]
            Log progress every N objects

        Yields
        ------
        str
            Object IDs in the dataset
        """
        pass

    def _check_object_id_to_tensor_cache(self, object_id: str):
        """Check if tensor is already cached."""
        return self.tensors.get(object_id, None)

    def _populate_object_id_to_tensor_cache(self, object_id: str):
        """Load tensor and populate cache."""
        data_torch = self._load_tensor_for_cache(object_id)
        self.tensors[object_id] = data_torch
        return data_torch

    def _object_id_to_tensor_cached(self, object_id: str):
        """
        Get tensor for object_id with caching support.

        Parameters
        ----------
        object_id : str
            The object_id requested

        Returns
        -------
        torch.Tensor
            The tensor for the object
        """
        start_time = time.monotonic_ns()

        if self.use_cache is False:
            return self._load_tensor_for_cache(object_id)

        data_torch = self._check_object_id_to_tensor_cache(object_id)
        if data_torch is not None:
            self._log_duration_tensorboard("cache_hit_s", start_time)
            return data_torch

        data_torch = self._populate_object_id_to_tensor_cache(object_id)
        self._log_duration_tensorboard("cache_miss_s", start_time)
        return data_torch

    @staticmethod
    def _determine_numprocs_preload():
        """Determine number of processes for preloading."""
        ##TO-DO: 50 is the optimized number for Hyak at UW
        ##This is totally file-system dependant and should
        ##be changed appropriately for other file-systems
        return 50

    def _preload_tensor_cache(self):
        """
        Preload all tensors in the dataset using multiple threads.
        """
        from concurrent.futures import ThreadPoolExecutor

        logger.info(f"Preloading {self.__class__.__name__} cache...")

        with ThreadPoolExecutor(max_workers=self._determine_numprocs_preload()) as executor:
            tensors = self._lazy_map_executor(executor, self.ids(log_every=1_000_000))

            start_time = time.monotonic_ns()
            for idx, (id, tensor) in enumerate(zip(self.ids(), tensors)):
                self.tensors[id] = tensor

                # Output timing every 1k tensors
                if idx % 1_000 == 0 and idx != 0:
                    self._log_duration_tensorboard("preload_1k_obj_s", start_time)
                    start_time = time.monotonic_ns()

    def _lazy_map_executor(self, executor: Executor, ids: Iterable[str]):
        """
        Lazy evaluation version of concurrent.futures.Executor.map().

        This limits memory usage during preloading by keeping only a small
        number of tensors in memory at once.

        Parameters
        ----------
        executor : concurrent.futures.Executor
            An executor for running futures
        ids : Iterable[str]
            An iterable list of object IDs

        Yields
        ------
        Iterator[torch.Tensor]
            An iterator over torch tensors, lazily loaded
        """
        from concurrent.futures import FIRST_COMPLETED, Future, wait

        from torch import Tensor

        max_futures = self._determine_numprocs_preload()
        queue: list[Future[Tensor]] = []
        in_progress: set[Future[Tensor]] = set()
        ids_iter = iter(ids)

        try:
            while True:
                for _ in range(max_futures - len(in_progress)):
                    id = next(ids_iter)
                    future = executor.submit(self._load_tensor_for_cache.__func__, self, id)  # type: ignore[attr-defined]
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

    def _log_duration_tensorboard(self, name: str, start_time: int):
        """
        Log a duration to tensorboardX if configured.

        Parameters
        ----------
        name : str
            The name of the scalar to log
        start_time : int
            Start time in nanoseconds from time.monotonic_ns()
        """
        now = time.monotonic_ns()
        name = f"{self.__class__.__name__}/" + name
        if self.tensorboardx_logger:
            since_tensorboard_start_us = (start_time - self.tensorboard_start_ns) / 1.0e3
            duration_s = (now - start_time) / 1.0e9
            self.tensorboardx_logger.add_scalar(name, duration_s, since_tensorboard_start_us)
