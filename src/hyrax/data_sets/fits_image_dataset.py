# ruff: noqa: D101, D102
"""
FitsImageDataSet is for if you have image data in a single directory and some sort of tabular catalog file.

At minimum, your tabular catalog **must** contain the following:

#. A unique ID column for each astronomical object you are interested in
#. A filename column containing the filename of the fits image file.
#. If you have multiple images with the same object ID, they must have separate rows in the catalog, one for each image. There must be a column describing the filter on the telescope that differentiates these objects

We recommend all your fits images be roughly the same size.

Setting up hyrax to use FitsImageDataSet works as follows in a notebook. The same configuration options can go
in a configuration file if you are running from the CLI

.. code-block:: python

    import hyrax
    h = hyrax.Hyrax()
    h.config["data_set"]["name"] = "FitsImageDataSet"
    h.config["general"]["data_dir"] = "/file/path/to/where/your/fits/files/are"

    # Location of your catalog file. Any file format supported by astropy.Table will work
    h.config["data_set"]["filter_catalog"] = "/file/path/to/your/catalog.fits"

    # Size in pixels to send to ML model. All images must be this size or larger on
    # both dimensions
    h.config["data_set"]["crop_to"] = (100,100)

    # This is good to simply attempt to construct the dataset. Once things are working you might try
    # to train or infer
    dataset = h.prepare()

This is the minimal setup that can work; however, there are several other configuration options you may need
to set depending on your usage.

The column names for the required columns are configurable. By default we use ``object_id``, ``filter``, and
``filename``; however, by setting ``h.config["data_set"]["object_id_column_name"]`` you can set the correct
name for your catalog file. ``h.config["data_set"]["filter_column_name"]`` and
``h.config["data_set"]["filename_column_name"]`` work in a corresponding manner.

If your dataset does not fit in memory on your system, we recommend setting
``h.config["data_set"]["use_cache"]`` and ``h.config["data_set"]["preload_cache"]`` to ``False``.
Both are ``True`` by default. The former caches all tensors read during an epoch into system RAM, with the
intent of speeding up later epochs of training if your disk has low bandwidth. The latter begins this process
of caching all tensors into system RAM in a background thread as soon as the ``FitsImageDataSet`` is
constructed, front-running the ``train`` or ``infer`` verb requesting tensors. The intent of this optimization
is to speed up the first epoch of training in the case where your disk has high latency. Both will result in
crashes if there is not enough room in your system RAM for the entire dataset.

If you need to truncate your dataset to fit in RAM, the easiest way is to select a small number of rows
from your original catalog file. FitsImageDataSet will only attempt to load images that exist in the catalog.

"""  # noqa: E501

import logging
import time
from collections.abc import Generator, Iterable
from concurrent.futures import Executor
from pathlib import Path
from threading import Thread
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from hyrax.config_utils import ConfigDict

from .data_set_registry import HyraxDataset, HyraxImageDataset

logger = logging.getLogger(__name__)

files_dict = dict[str, dict[str, str]]


class FitsImageDataSet(HyraxDataset, HyraxImageDataset, Dataset):
    """
    Dataset for Fits Images, typically cutouts.
    """

    _called_from_test = False

    def __init__(self, config: ConfigDict, data_location=None):
        """
        .. py:method:: __init__

        Initialize a FitsImageDataSet

        Most work is done in ``_init_from_path`` and functions it calls in order to allow
        subclasses to override behavior.

        Parameters
        ----------
        config : ConfigDict
            Nested configuration dictionary for hyrax
        data_location : Optional[Union[Path, str]]
            The directory location of the data that this dataset class will access
        """

        self._config = config

        self.use_cache = config["data_set"]["use_cache"]

        self.set_function_transform()

        self.object_id_column_name = (
            config["data_set"]["object_id_column_name"]
            if config["data_set"]["object_id_column_name"]
            else "object_id"
        )
        self.filter_column_name = (
            config["data_set"]["filter_column_name"] if config["data_set"]["filter_column_name"] else "filter"
        )
        self.filename_column_name = (
            config["data_set"]["filename_column_name"]
            if config["data_set"]["filename_column_name"]
            else "filename"
        )

        self._init_from_path(data_location)

        # Relies on self.filters_ref and self.filter_catalog_table which are both determined
        # inside _init_from_path()
        logger.debug("Preparing Metadata")
        metadata = self._prepare_metadata()
        super().__init__(config, metadata)

        self._before_preload()

        if config["data_set"]["preload_cache"] and self.use_cache:
            self.preload_thread = Thread(
                name=f"{self.__class__.__name__}-preload-tensor-cache",
                daemon=True,
                # Note we are passing only the function and self explicitly to the thread
                # This ensures the current object is in shared thread memory.
                target=self._preload_tensor_cache.__func__,  # type: ignore[attr-defined]
                args=(self,),
            )
            self.preload_thread.start()

    def _init_from_path(self, path: Union[Path, str]):
        """__init__ helper. Initialize an HSC data set from a path. This involves several filesystem scan
        operations and will ultimately open and read the header info of every fits file in the given directory

        Parameters
        ----------
        path : Union[Path, str]
            Path or string specifying the directory path that is the root of all filenames in the
            catalog table
        """
        from torch import Tensor

        self.path = path

        # This is common code
        filter_catalog = None
        if self.config["data_set"]["filter_catalog"]:
            filter_catalog = Path(self.config["data_set"]["filter_catalog"])

        self.filter_catalog_table = self._read_filter_catalog(filter_catalog)
        self.files = self._parse_filter_catalog(self.filter_catalog_table)
        if self.files is None:
            msg = "Cannot continue without files. Please ensure the table passed in "
            msg += "config['data_set']['filter_catalog'] is well formed. It should minimally be "
            msg += f"a table readable by Astropy's Table.read() with columns: {self.object_id_column_name}, "
            msg += f"{self.filename_column_name}, and {self.filter_column_name}. This may also occur because "
            msg += "of a misimplemented subclass"
            raise RuntimeError(msg)

        first_filter_dict = next(iter(self.files.values()))
        self.num_filters = len(first_filter_dict)

        self._set_crop_transform()

        self.tensors: dict[str, Tensor] = {}
        self.tensorboard_start_ns = time.monotonic_ns()
        self.tensorboardx_logger = None

        logger.info(f"FitsImageDataSet has {len(self)} objects")

    def _set_crop_transform(self):
        """
        Returns the crop transform on the image

        If overriden, subclass must:
        1) set self.cutout_shape to a tuple of ints representing the size of the cutouts that will be
        returned at some point in the init flow.

        2) Update the crop tranform using self.set_crop_transform() from the HyraxImageDataset mixin
        """

        self.cutout_shape = self.config["data_set"]["crop_to"] if self.config["data_set"]["crop_to"] else None
        self.set_crop_transform()

    def _read_filter_catalog(self, filter_catalog_path: Optional[Path]):
        from astropy.table import Table

        if filter_catalog_path is None:
            msg = "Must provide a filter catalog in config['data_set']['filter_catalog']"
            raise RuntimeError(msg)

        if not filter_catalog_path.exists():
            msg = f"Filter catalog file {filter_catalog_path} given in config does not exist."
            raise RuntimeError(msg)

        table = Table.read(filter_catalog_path)
        colnames = table.colnames

        object_id_missing = self.object_id_column_name not in colnames
        filename_missing = self.filename_column_name not in colnames
        filter_missing = self.filter_column_name not in colnames

        if object_id_missing:
            msg = f"Filter catalog file {filter_catalog_path} has no column '{self.object_id_column_name}'"
            raise RuntimeError(msg)

        if filename_missing:
            msg = f"Filter catalog file {filter_catalog_path} has no column '{self.filename_column_name}'"
            raise RuntimeError(msg)

        if filter_missing:
            msg = f"Filter catalog file {filter_catalog_path} has no column '{self.filter_column_name}'. "
            logger.warning(msg)

            _, counts = np.unique(table[self.object_id_column_name], return_counts=True)
            if np.max(counts) == 1:
                msg = "Object IDs are unique, filling in the same filter value across all objects"
                logger.warning(msg)
                table.add_column(np.full(len(table), "Unknown_filter"), name=self.filter_column_name)
            else:
                msg = "Object IDs are not unique. you must add a 'filter' column to your table or name "
                msg += "the appropriate column by setting config['data_set']['filter_column_name']"
                raise RuntimeError(msg)

        table.add_index(self.object_id_column_name)

        if not filter_missing:
            table.add_index(self.filter_column_name)

        return table

    def _parse_filter_catalog(self, table) -> None:
        """Sets self.files by parsing the catalog.

        Subclasses may override this function to control parsing of the table more directly, but the
        overriding class must create the files dict which has type dict[object_id -> dict[filter -> filename]]
        with object_id, filter, and filename all strings.  In the case of no filter distinction, a single
        flag value may be used for the filter dict keys in the inner dicts.

        Parameters
        ----------
        table : Table
            The catalog we read in

        """
        filter_catalog: files_dict = {}

        for row in table:
            object_id = str(row[self.object_id_column_name])
            filter = row[self.filter_column_name]
            filename = row[self.filename_column_name]

            # Insert into the filter catalog.
            if object_id not in filter_catalog:
                filter_catalog[object_id] = {}
            filter_catalog[object_id][filter] = filename

        return filter_catalog

    def _before_preload(self) -> None:
        # Provided so subclasses can make edits to the class after full initialization
        # but before the cache preload thread starts iterating over the datastructure and
        # fetching
        pass

    def _prepare_metadata(self):
        # This happens when filter_catalog_table is injected in unit tests
        if FitsImageDataSet._called_from_test:
            return None

        if self.filter_catalog_table is None:
            return None

        # Get all object_ids in enumeration order
        sorted_object_ids = np.array([int(id) for id in self.ids()])

        # Filter for a single reference filter to deduplicate object_id rows
        first_filter_dict = next(iter(self.files.values()))
        first_filter = next(iter(first_filter_dict))
        mask = self.filter_catalog_table[self.filter_column_name] == first_filter
        filter_catalog_table_dedup = self.filter_catalog_table[mask]

        # Build fast lookup from object_id to row index
        id_to_index = {oid: i for i, oid in enumerate(filter_catalog_table_dedup[self.object_id_column_name])}

        # Extract rows in the desired order
        try:
            row_indices = [id_to_index[oid] for oid in sorted_object_ids]
        except KeyError as e:
            missing_id = e.args[0]
            logger.error(f"Object ID {missing_id} not found in filtered metadata table.")
            raise

        metadata = filter_catalog_table_dedup[row_indices]

        # Filter for the appropriate columns
        colnames = list(self.filter_catalog_table.colnames)
        # colnames.remove(self.filename_column_name)
        # colnames.remove(self.filter_column_name)

        logger.debug("Finished preparing metadata")
        return metadata[colnames]

    def shape(self) -> tuple[int, int, int]:
        """Shape of the individual cutouts this will give to a model

        Returns
        -------
        tuple[int,int,int]
            Tuple describing the dimensions of the 3 dimensional tensor handed back to models
            The first index is the number of filters
            The second index is the width of each image
            The third index is the height of each image
        """
        return (self.num_filters, self.cutout_shape[0], self.cutout_shape[1])

    def __len__(self) -> int:
        """Returns number of objects in this loader

        Returns
        -------
        int
            number of objects in this data loader
        """
        return len(self.files)

    def get_object_id(self, idx: int) -> str:
        """Get the object ID at the given index

        Parameters
        ----------
        idx : int
            Index of the object ID to return

        Returns
        -------
        str
            The object ID at the given index
        """
        if idx >= len(self.files) or idx < 0:
            raise IndexError("Index out of range")

        # Use the list of object IDs for explicit indexing
        return list(self.files.keys())[idx]

    def get_image(self, idx: int):
        """Get the image at the given index as a PyTorch Tensor.

        Parameters
        ----------
        idx : int
            Index of the image to return

        Returns
        -------
        torch.Tensor
            The image at the given index as a PyTorch Tensor.
        """
        object_id = self.get_object_id(idx)
        return self._object_id_to_tensor(object_id)

    def __getitem__(self, idx: int):
        if idx >= len(self.files) or idx < 0:
            raise IndexError

        object_id = self.get_object_id(idx)

        return {
            "data": {
                "object_id": object_id,
                "image": self.get_image(idx),
                "index": idx,
            },
            "object_id": object_id,
        }

    def __contains__(self, object_id: str) -> bool:
        """Allows you to do `object_id in dataset` queries. Used by testing code.

        Parameters
        ----------
        object_id : str
            The object ID you'd like to know if is in the dataset

        Returns
        -------
        bool
            True of the object_id given is in the data set
        """
        return object_id in list(self.files.keys())

    def _get_file(self, index: int) -> Path:
        """Private indexing method across all files.

        Returns the file path corresponding to the given index.

        The index is zero-based and defined in the same manner as the total order of _all_files() and
        _object_files() iterator. Useful if you have an np.array() or list built from _all_files() and you
        need to select an individual item.

        Only valid after self.object_ids, self.files, self.path, and self.num_filters have been initialized
        in __init__

        Parameters
        ----------
        index : int
            Index, see above for order semantics

        Returns
        -------
        Path
            The path to the file
        """
        object_index = int(index / self.num_filters)
        object_id = list(self.files.keys())[object_index]
        filters = self.files[object_id]
        filter_names = sorted(list(filters))
        filter = filter_names[index % self.num_filters]
        return self._file_to_path(filters[filter])

    def ids(self, log_every=None) -> Generator[str]:
        """Public read-only iterator over all object_ids that enforces a strict total order across
        objects. Will not work prior to self.files initialization in __init__

        Yields
        ------
        Iterator[str]
            Object IDs currently in the dataset
        """
        log = log_every is not None and isinstance(log_every, int)
        for index, object_id in enumerate(self.files):
            if log and index != 0 and index % log_every == 0:
                logger.info(f"Processed {index + 1} objects")
            yield str(object_id)
        else:
            if log:
                logger.info(f"Processed {index + 1} objects")

    def _all_files(self):
        """
        Private read-only iterator over all files that enforces a strict total order across
        objects and filters. Will not work prior to self.files, and self.path initialization in __init__

        Yields
        ------
        Path
            The path to the file.
        """
        for object_id in self.ids():
            for filename in self._object_files(object_id):
                yield filename

    def _filter_filename(self, object_id):
        """
        Private read-only iterator over all files for a given object. This enforces a strict total order
        across filters. Will not work prior to self.files initialization in __init__

        Yields
        ------
        filter_name, file name
            The name of a filter and the file name for the fits file.
            The file name is relative to self.path
        """
        filters = self.files[object_id]
        filter_names = sorted(list(filters))
        for filter_name in filter_names:
            yield filter_name, filters[filter_name]

    def _object_files(self, object_id):
        """
        Private read-only iterator over all files for a given object. This enforces a strict total order
        across filters. Will not work prior to self.files, and self.path initialization in __init__

        Yields
        ------
        Path
            The path to the file.
        """
        for _, filename in self._filter_filename(object_id):
            yield self._file_to_path(filename)

    def _file_to_path(self, filename: str) -> Path:
        """Turns a filename into a full path suitable for open. Equivalent to:

        `Path(self.path) / Path(filename)`

        Parameters
        ----------
        filename : str
            The filename string

        Returns
        -------
        Path
            A full path that is openable.
        """
        return Path(self.path) / Path(filename)

    @staticmethod
    def _determine_numprocs_preload():
        # This is hardcoded to a reasonable value for hyak
        # TODO: Unify this function and _determine_numprocs(). Ideally we would have
        # either a multiprocessing.Pool or concurrent.futures.Executor interface taking
        # an i/o bound callable which returns a number of bytes read. This reusable
        # component would titrate the number of worker threads/processes to achieve a
        # maximal throughput as measured by returns from the callable vs wall-clock time.
        return 50

    def _preload_tensor_cache(self):
        """
        When preloading the tensor cache is configured, this is called on a separate thread by __init__()
        to perform a preload of every tensor in the dataset.
        """
        from concurrent.futures import ThreadPoolExecutor

        logger.info("Preloading FitsImageDataSet cache...")

        with ThreadPoolExecutor(max_workers=FitsImageDataSet._determine_numprocs_preload()) as executor:
            tensors = self._lazy_map_executor(executor, self.ids(log_every=1_000_000))

            start_time = time.monotonic_ns()
            for idx, (id, tensor) in enumerate(zip(self.ids(), tensors)):
                self.tensors[id] = tensor

                # Output timing every 1k tensors
                if idx % 1_000 == 0 and idx != 0:
                    self._log_duration_tensorboard("preload_1k_obj_s", start_time)
                    start_time = time.monotonic_ns()

    def _lazy_map_executor(self, executor: Executor, ids: Iterable[str]):
        """This is a version of concurrent.futures.Executor map() which lazily evaluates the iterator passed
        We do this because we do not want all of the tensors to remain in memory during pre-loading. We would
        prefer a smaller set of in-flight tensors.

        The total number of in progress jobs is set at FitsImageDataSet._determine_numprocs().

        The total number of tensors is slightly greater than that owing to out-of-order execution.

        This approach was copied from:
        https://gist.github.com/CallumJHays/0841c5fdb7b2774d2a0b9b8233689761

        Parameters
        ----------
        executor : concurrent.futures.Executor
            An executour for running our futures
        work_fn : Callable[[str], torch.Tensor]
            The function that makes tensors out of object_ids
        ids : Iterable[str]
            An iterable list of object IDs.

        Yields
        ------
        Iterator[torch.Tensor]
            An iterator over torch tensors, lazily loaded by running the work_fn as needed.
        """
        from concurrent.futures import FIRST_COMPLETED, Future, wait

        from torch import Tensor

        max_futures = FitsImageDataSet._determine_numprocs_preload()
        queue: list[Future[Tensor]] = []
        in_progress: set[Future[Tensor]] = set()
        ids_iter = iter(ids)

        try:
            while True:
                for _ in range(max_futures - len(in_progress)):
                    id = next(ids_iter)
                    future = executor.submit(self._read_object_id.__func__, self, id)  # type: ignore[attr-defined]
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
        """Log a duration to tensorboardX. NOOP if no tensorboard logger configured

        The time logged is a floating point number of seconds derived from integer
        monotonic nanosecond measurements. time.monotonic_ns() is used for the current time

        The step number for the scalar series is an integer number of microseonds.

        Parameters
        ----------
        name : str
            The name of the scalar to log to tensorboard
        start_time : int
            integer number of nanoseconds. Should be from time.monotonic_ns() when the duration started

        """
        now = time.monotonic_ns()
        name = f"{self.__class__.__name__}/" + name
        if self.tensorboardx_logger:
            since_tensorboard_start_us = (start_time - self.tensorboard_start_ns) / 1.0e3

            duration_s = (now - start_time) / 1.0e9
            self.tensorboardx_logger.add_scalar(name, duration_s, since_tensorboard_start_us)

    def _check_object_id_to_tensor_cache(self, object_id: str):
        return self.tensors.get(object_id, None)

    def _populate_object_id_to_tensor_cache(self, object_id: str):
        data_torch = self._read_object_id(object_id)
        self.tensors[object_id] = data_torch
        return data_torch

    def _read_object_id(self, object_id: str):
        from astropy.io import fits

        start_time = time.monotonic_ns()

        # Read all the files corresponding to this object
        data = []

        for filepath in self._object_files(object_id):
            file_start_time = time.monotonic_ns()
            raw_data = fits.getdata(filepath, memmap=False)
            data.append(raw_data)
            self._log_duration_tensorboard("file_read_time_s", file_start_time)

        self._log_duration_tensorboard("object_read_time_s", start_time)

        data_torch = self._convert_to_torch(data)
        self._log_duration_tensorboard("object_total_read_time_s", start_time)
        return data_torch

    def _convert_to_torch(self, data: list[npt.ArrayLike]):
        from torch import from_numpy

        start_time = time.monotonic_ns()

        # Push all the filter data into a tensor object
        data_np = np.array(data)
        data_torch = from_numpy(data_np.astype(np.float32))

        # Apply our transform stack
        data_torch = self.transform(data_torch) if self.transform is not None else data_torch

        self._log_duration_tensorboard("object_convert_tensor_time_s", start_time)
        return data_torch

    # TODO: Performance Change when files are read/cache pytorch tensors?
    #
    # This function loads from a file every time __getitem__ is called
    # Do we want to pre-cache these into memory in init?
    # Do we want to memoize them on first __getitem__ call?
    #
    # For now we just do it the naive way
    def _object_id_to_tensor(self, object_id: str):
        """Converts an object_id to a pytorch tensor with dimensions (self.num_filters, self.cutout_shape[0],
        self.cutout_shape[1]). This is done by reading the file and slicing away any excess pixels at the
        far corners of the image from (0,0).

        The current implementation reads the files once the first time they are accessed, and then
        keeps them in a dict for future accesses.

        Parameters
        ----------
        object_id : str
            The object_id requested

        Returns
        -------
        torch.Tensor
            A tensor with dimension (self.num_filters, self.cutout_shape[0], self.cutout_shape[1])
        """
        start_time = time.monotonic_ns()

        if self.use_cache is False:
            return self._read_object_id(object_id)

        data_torch = self._check_object_id_to_tensor_cache(object_id)
        if data_torch is not None:
            self._log_duration_tensorboard("cache_hit_s", start_time)
            return data_torch

        data_torch = self._populate_object_id_to_tensor_cache(object_id)
        self._log_duration_tensorboard("cache_miss_s", start_time)
        return data_torch
