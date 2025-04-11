# ruff: noqa: D101, D102

import datetime
import logging
import multiprocessing
import os
import re
import resource
import time
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import Executor
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.table import Table
from schwimmbad import MultiPool
from torch import Tensor, from_numpy
from torch.utils.data import Dataset
from torchvision.transforms.v2 import CenterCrop, Compose, Lambda

from hyrax.config_utils import ConfigDict
from hyrax.download import Downloader
from hyrax.downloadCutout.downloadCutout import (
    parse_bool,
    parse_degree,
    parse_latitude,
    parse_longitude,
    parse_rerun,
    parse_tract_opt,
    parse_type,
)

from .data_set_registry import HyraxDataset

logger = logging.getLogger(__name__)
dim_dict = dict[str, list[tuple[int, int]]]
files_dict = dict[str, dict[str, str]]


class HSCDataSet(HyraxDataset, Dataset):
    _called_from_test = False

    def __init__(self, config: ConfigDict):
        crop_to = config["data_set"]["crop_to"]
        filters = config["data_set"]["filters"]
        transform_str = config["data_set"]["transform"]
        self.use_cache = config["data_set"]["use_cache"]

        if transform_str:
            transform_func = self._get_np_function(transform_str)
            transform = Lambda(lambd=transform_func)
        else:
            transform = None

        # Note "rebuild_manifest" is not a config, its a hack for rebuild_manifest mode
        # to ensure we don't use the manifest we believe is corrupt.
        rebuild_manifest = config["rebuild_manifest"] if "rebuild_manifest" in config else False  # noqa: SIM401

        if config["data_set"]["filter_catalog"]:
            filter_catalog = Path(config["data_set"]["filter_catalog"])
        elif not rebuild_manifest:
            filter_catalog = Path(config["general"]["data_dir"]) / Downloader.MANIFEST_FILE_NAME
            if not filter_catalog.exists():
                filter_catalog = None
        else:
            filter_catalog = None

        self._init_from_path(
            config["general"]["data_dir"],
            transform=transform,
            cutout_shape=crop_to if crop_to else None,
            filters=filters if filters else None,
            filter_catalog=filter_catalog,
        )

        # Relies on self.filters_ref and self.filter_catalog_table which are both determined
        # inside _init_from_path()
        metadata = self._prepare_metadata()
        super().__init__(config, metadata)

        if config["data_set"]["preload_cache"] and self.use_cache:
            self.preload_thread = Thread(
                name="HSCDataSet-preload-tensor-cache",
                daemon=True,
                # Note we are passing only the function and self explicitly to the thread
                # This ensures the current object is in shared thread memory.
                target=self._preload_tensor_cache.__func__,  # type: ignore[attr-defined]
                args=(self,),
            )
            self.preload_thread.start()

    def _get_np_function(self, transform_str: str) -> Callable[..., Any]:
        """
        _get_np_function. Returns the numpy mathematical function that the
        supplied string maps to; or raises an error if the supplied string
        cannot be mapped to a function.

        Parameters
        ----------
        transform_str: str
            The string to me mapped to a numpy function
        """

        try:
            func: Callable[..., Any] = getattr(np, transform_str)
            if callable(func):
                return func
        except AttributeError as err:
            msg = f"{transform_str} is not a valid numpy function.\n"
            msg += "The string passed to the transform variable needs to be a numpy function"
            raise RuntimeError(msg) from err

    def _init_from_path(
        self,
        path: Union[Path, str],
        *,
        transform=None,
        cutout_shape: Optional[tuple[int, int]] = None,
        filters: Optional[list[str]] = None,
        filter_catalog: Optional[Path] = None,
    ):
        """__init__ helper. Initialize an HSC data set from a path. This involves several filesystem scan
        operations and will ultimately open and read the header info of every fits file in the given directory

        Parameters
        ----------
        path : Union[Path, str]
            Path or string specifying the directory path to scan. It is expected that all files will
            be flat in this directory
        transform : torchvision.transforms.v2.Transform, optional
            Transformation to apply to every image in the dataset, by default None
        cutout_shape: tuple[int,int], optional
            Forces all cutouts to be a particular pixel size. If this size is larger than the pixel dimension
            of particular cutouts on the filesystem, those objects are dropped from the data set.
        filters: list[str], optional
            Forces all cutout tensors provided to be from the list of HSC filters provided. If provided, any
            cutouts which do not have fits files corresponding to every filter in the list will be dropped
            from the data set. Defaults to None. If not provided, the filters available on the filesystem for
            the first object in the directory will be used.
        filter_catalog: Path, optional
            Path to a .fits file which specifies objects and or files to use directly, bypassing the default
            of attempting to use every file in the path.
            Columns for this fits file are object_id (required), filter (optional), filename (optional), and
            dims (optional tuple of x/y pixel size of images).
             - Filenames must be relative to the path provided to this function.
             - When filters and filenames are both provided, initialization skips a directory listing, which
               can provide better performance on large datasets.
             - When filters, filenames, and dims are specified we also skip opening the files to get
               the dimensions. This can also provide better performance on large datasets.
        """
        self.path = path
        self.transform = transform

        self.filter_catalog_table = self._read_filter_catalog(filter_catalog)

        self.filter_catalog = (
            None
            if self.filter_catalog_table is None
            else self._parse_filter_catalog(self.filter_catalog_table)
        )

        if isinstance(self.filter_catalog, tuple):
            self.files = self.filter_catalog[0]
            self.dims = self.filter_catalog[1]
        elif isinstance(self.filter_catalog, dict):
            self.files = self.filter_catalog
            self.dims = self._scan_file_dimensions()
        else:
            self.files = self._scan_file_names(filters)
            self.dims = self._scan_file_dimensions()

        # If no filters provided, we choose the first file in the dict as the prototypical set of filters
        # Any objects lacking this full set of filters will be pruned by _prune_objects
        self.filters_ref = list(list(self.files.values())[0]) if filters is None else filters

        self.num_filters = len(self.filters_ref)

        self.pruned_objects: dict[str, str] = {}
        self._prune_objects(self.filters_ref, cutout_shape)

        self.cutout_shape = self._check_file_dimensions() if cutout_shape is None else cutout_shape

        # Set up our default transform to center-crop the image to the common size before
        # Applying any transforms we were passed.
        crop = CenterCrop(size=self.cutout_shape)
        self.transform = Compose([crop, self.transform]) if self.transform is not None else crop

        self.tensors: dict[str, Tensor] = {}
        self.tensorboard_start_ns = time.monotonic_ns()
        self.tensorboardx_logger = None

        logger.info(f"HSC Data set loader has {len(self)} objects")

    def _scan_file_names(self, filters: Optional[list[str]] = None) -> files_dict:
        """Class initialization helper

        Parameters
        ----------
        filters : list[str], optional
            If passed, only these filters will be scanned for from the data files. Defaults to None, which
            corresponds to the standard set of filters ["HSC-G","HSC-R","HSC-I","HSC-Z","HSC-Y"].

        Returns
        -------
        dict[str,dict[str,str]]
            Nested dictionary where the first level maps object_id -> dict, and the second level maps
            filter_name -> file name. Corresponds to self.files
        """

        logger.info(f"Scanning files in directory {self.path}")

        object_id_regex = r"[0-9]{17}"
        filter_regex = r"HSC-[GRIZY]" if filters is None else "|".join(filters)
        full_regex = f"({object_id_regex})_.*_({filter_regex}).fits"

        files: files_dict = {}
        # Go scan the path for object ID's so we have a list.
        for index, filepath in enumerate(Path(self.path).iterdir()):
            filename = filepath.name

            # If we are filtering based off a user-provided catalog of object ids, Filter out any
            # objects_ids not in the catalog. Do this before regex match for speed of discarding
            # irrelevant files.
            if isinstance(self.filter_catalog, list) and filename[:17] not in self.filter_catalog:
                continue

            m = re.match(full_regex, filename)
            # Skip files that don't allow us to extract both object_id and filter
            if m is None:
                continue

            object_id = m[1]
            filter = m[2]

            if files.get(object_id) is None:
                files[object_id] = {}

            if files[object_id].get(filter) is None:
                files[object_id][filter] = filename
            else:
                msg = f"Duplicate object ID {object_id} detected.\n"
                msg += f"File {filename} conflicts with already scanned file {files[object_id][filter]} "
                msg += "and will not be included in the data set."
                logger.error(msg)

            if index != 0 and index % 1_000_000 == 0:
                logger.info(f"Processed {index} files.")
        else:
            logger.info(f"Processed {index + 1} files")

        return files

    def _read_filter_catalog(self, filter_catalog_path: Optional[Path]) -> Table:
        if filter_catalog_path is None:
            return None

        if not filter_catalog_path.exists():
            logger.error(f"Filter catalog file {filter_catalog_path} given in config does not exist.")
            return None

        table = Table.read(filter_catalog_path, format="fits")
        colnames = table.colnames

        if "object_id" not in colnames:
            logger.error(f"Filter catalog file {filter_catalog_path} has no column object_id")
            return None

        table.add_index("object_id")
        table.add_index("filter")

        if ("filter" not in colnames) ^ ("filename" not in colnames):
            msg = f"Filter catalog file {filter_catalog_path} provides one of filters or filenames "
            msg += "without the other. Filesystem scan will still occur without both defined."
            logger.warning(msg)

        return table

    def _parse_filter_catalog(
        self, table: Table
    ) -> Optional[Union[list[str], files_dict, tuple[files_dict, dim_dict]]]:
        colnames = table.colnames
        # We are dealing with just a list of object_ids
        if "filter" not in colnames and "filename" not in colnames:
            return list(table["object_id"])

        # Or a table that lacks both filter and filename
        elif "filter" not in colnames or "filename" not in colnames:
            return list(set(table["object_id"]))

        # We have filter and filename defined so we can assemble the catalog at file level.
        filter_catalog: files_dict = {}
        if "dim" in colnames:
            dim_catalog: dim_dict = {}

        for row in table:
            object_id = str(row["object_id"])
            filter = row["filter"]
            filename = row["filename"]
            if "dim" in colnames:
                dim = tuple(row["dim"])

            # Skip over any files that are marked as didn't download.
            # or have a dimension listed less than 1px x 1px
            if filename == "Attempted" or min(dim) < 1:
                continue

            # Insert into the filter catalog.
            if object_id not in filter_catalog:
                filter_catalog[object_id] = {}
            filter_catalog[object_id][filter] = filename

            # Dimension is optional, insert into dimension catalog.
            if "dim" in colnames:
                if object_id not in dim_catalog:
                    dim_catalog[object_id] = []
                dim_catalog[object_id].append(dim)

        return (filter_catalog, dim_catalog) if "dim" in colnames else filter_catalog

    @staticmethod
    def _determine_numprocs() -> int:
        # Figure out how many CPUs we are allowed to use
        cpu_count = None
        sched_getaffinity = getattr(os, "sched_getaffinity", None)

        if sched_getaffinity:
            cpu_count = len(sched_getaffinity(0))
        elif multiprocessing:
            cpu_count = multiprocessing.cpu_count()
        else:
            cpu_count = 1

        # Ideally we would use ~75 processes per CPU to attempt to saturate
        # I/O bandwidth using a small number of CPUs.
        numproc = 1 if HSCDataSet._called_from_test else 75 * cpu_count
        numproc = HSCDataSet._fixup_limit(
            numproc,
            resource.RLIMIT_NOFILE,
            lambda proc: int(4 * proc + 10),
            lambda nofile: int((nofile - 10) / 4),
        )

        numproc = HSCDataSet._fixup_limit(
            numproc, resource.RLIMIT_NPROC, lambda proc: proc, lambda proc: proc
        )
        return numproc

    @staticmethod
    def _fixup_limit(nproc: int, res, est_limit, est_procs) -> int:
        # If launching this many processes would trigger other resource limits, work around them
        limit_soft, limit_hard = resource.getrlimit(res)

        # If we would violate the hard limit, calculate the number of processes that wouldn't
        # violate the limit
        if limit_hard < est_limit(nproc):
            nproc = est_procs(limit_hard)

        # If we would violate the soft limit, attempt to change it, leaving the hard limit alone
        try:
            if limit_soft < est_limit(nproc):
                resource.setrlimit(res, (est_limit(nproc), limit_hard))
        finally:
            # If the change doesn't take, then reduce the number of processes again
            limit_soft, limit_hard = resource.getrlimit(res)
            if limit_soft < est_limit(nproc):
                nproc = est_procs(limit_soft)

        return nproc

    def _scan_file_dimensions(self) -> dim_dict:
        # Scan the filesystem to get the widths and heights of all images into a dict
        logger.info("Scanning for dimensions...")

        retval = {}
        with MultiPool(processes=HSCDataSet._determine_numprocs()) as pool:
            args = (
                (object_id, list(self._object_files(object_id)))
                for object_id in self.ids(log_every=1_000_000)
            )
            retval = dict(pool.imap(self._scan_file_dimension, args, chunksize=1000))
        return retval

    @staticmethod
    def _scan_file_dimension(processing_unit: tuple[str, list[str]]) -> tuple[str, list[tuple[int, int]]]:
        object_id, filenames = processing_unit
        return (object_id, [HSCDataSet._fits_file_dims(filepath) for filepath in filenames])

    @staticmethod
    def _fits_file_dims(filepath) -> tuple[int, int]:
        try:
            with fits.open(filepath) as hdul:
                return (hdul[1].shape[0], hdul[1].shape[1])
        except OSError:
            return (0, 0)

    def _prune_objects(self, filters_ref: list[str], cutout_shape: Optional[tuple[int, int]]):
        """Class initialization helper. Prunes objects from the list of objects.

        1) Removes any objects which do not have all the filters specified in filters_ref
        2) If a cutout_shape was provided in the constructor, prunes files that are too small
           for the chosen cutout size

        This function deletes from self.files and self.dims via _prune_object

        Parameters
        ----------
        files : dict[str,dict[str,str]]
            Nested dictionary where the first level maps object_id -> dict, and the second level maps
            filter_name -> file name. This is created by _scan_files()

        filters_ref : list[str]
            List of the filter names

        cutout_shape: : tuple[int, int]
            Cutout shape tuple provided from constructor
        """
        filters_ref = sorted(filters_ref)
        self.prune_count = 0
        for index, (object_id, filters_unsorted) in enumerate(self.files.items()):
            # Drop objects with missing filters
            filter_intersect = sorted([filter for filter in filters_unsorted if filter in filters_ref])
            if filter_intersect != filters_ref:
                msg = f"HSCDataSet in {self.path} has the wrong group of filters for object {object_id}."
                self._mark_for_prune(object_id, msg)
                logger.info(f"Filters for object {object_id} were {filters_unsorted}")
                logger.debug(f"Reference filters were {filters_ref}")

            elif cutout_shape is not None:
                # Drop objects that can't meet the cutout size provided
                for shape in self.dims[object_id]:
                    if shape[0] < cutout_shape[0] or shape[1] < cutout_shape[1]:
                        msg = f"A file for object {object_id} has shape ({shape[0]}px, {shape[1]}px)"
                        msg += " this is too small for the given cutout size of "
                        msg += f"({cutout_shape[0]}px, {cutout_shape[1]}px)"
                        self._mark_for_prune(object_id, msg)
                        break

                # Drop objects where the cutouts are not the same size
                first_shape = None
                for shape in self.dims[object_id]:
                    first_shape = shape if first_shape is None else first_shape
                    if shape != first_shape:
                        msg = f"The first filter for object {object_id} has a shape of "
                        msg += f"({first_shape[0]}px,{first_shape[1]}px) another filter has shape of"
                        msg += f"({shape[0]}px,{shape[1]}px)"
                        self._mark_for_prune(object_id, msg)
                        break

                # Drop objects where parsing the filenames does not reveal the object IDs
                for filter, filepath in filters_unsorted.items():
                    filename = Path(filepath).name
                    # Check beginning of filename vs object_id
                    if filename[:17] != object_id:
                        msg = f"Filter {filter} for object id {object_id} has filename {filepath} listed"
                        msg += "The filename does not match the object_id, and the filter_catalog or "
                        msg += "manifest is likely corrupt."
                        self._mark_for_prune(object_id, msg)
                        break

            if index != 0 and index % 1_000_000 == 0:
                logger.info(f"Processed {index} objects for pruning")
        else:
            logger.info(f"Processed {index + 1} objects for pruning")

        # Prune marked objects
        for object_id, reason in self.pruned_objects.items():
            self._prune_object(object_id, reason)

        # Log about the pruning process
        pre_prune_object_count = len(self.files) + self.prune_count
        prune_fraction = self.prune_count / pre_prune_object_count
        if prune_fraction > 0.05:
            logger.error("Greater than 5% of objects in the data directory were pruned.")
        elif prune_fraction > 0.01:
            logger.warning("Greater than 1% of objects in the data directory were pruned.")

        if self.prune_count > 0:
            logger.info(f"Pruned {self.prune_count} out of {pre_prune_object_count} objects")

    def _mark_for_prune(self, object_id, reason):
        self.pruned_objects[object_id] = reason

    def _prune_object(self, object_id, reason: str):
        logger.warning(reason)
        logger.warning(f"Dropping object {object_id} from the dataset")

        del self.files[object_id]
        del self.dims[object_id]
        self.prune_count += 1

    def _check_file_dimensions(self) -> tuple[int, int]:
        """Class initialization helper. Find the maximal pixel size that all images can support

        It is assumed that all the cutouts will be of very similar size; however, HSC's cutout
        server does not return exactly the same number of pixels for every query, even when it
        is given the same angular spread for every cutout.

        Machine learning models expect all images to be the same size.

        This function warns on significant differences (>2px) on any dimension between the largest
        and smallest images.

        Returns
        -------
        tuple(int,int)
            The minimum width and height in pixels of the entire dataset. In other words: the maximal image
            size in pixels that can be generated from ALL cutout images via cropping.
        """
        logger.info("Checking file dimensions to determine standard cutout size...")

        # Find the maximal cutout size that all images can support
        all_widths = [shape[0] for shape_list in self.dims.values() for shape in shape_list]
        all_heights = [shape[1] for shape_list in self.dims.values() for shape in shape_list]
        all_dimensions = all_widths + all_heights
        cutout_height = np.min(all_dimensions)
        cutout_width = cutout_height

        if (
            np.abs(cutout_width - np.mean(all_widths)) > 1
            or np.abs(cutout_height - np.mean(all_heights)) > 1
            or np.abs(np.max(all_widths) - np.mean(all_widths)) > 1
            or np.abs(np.max(all_heights) - np.mean(all_heights)) > 1
        ):
            msg = "Some images differ from the mean width or height of all images by more than 1px\n"
            msg += f"Images will be cropped to ({cutout_width}px, {cutout_height}px)\n"
            try:
                min_width_file = self._get_file(int(np.argmin(all_widths)))
                min_height_file = self._get_file(int(np.argmin(all_heights)))
                msg += f"See {min_width_file} for an example image of width {cutout_width}px\n"
                msg += f"See {min_height_file} for an example image of height {cutout_height}px"
            finally:
                logger.warning(msg)

        if min(cutout_height, cutout_width) < 1:
            msg = "Automatic determination found an absurd dimension of "
            msg += f"({cutout_width}px, {cutout_height}px)\n"
            msg += "Please either correct the data source or set a static cutout size with the \n"
            msg += "crop_to configuration in the [data_set] section of your hyrax config.\n"
            raise RuntimeError(msg)

        return cutout_width, cutout_height

    def _rebuild_manifest(self, config):
        if self.filter_catalog:
            raise RuntimeError("Cannot rebuild manifest. Set the filter_catalog=false and rerun")

        logger.info("Reading in catalog file... ")
        location_table = Downloader.filterfits(
            Path(config["download"]["fits_file"]).expanduser().resolve(), ["object_id", "ra", "dec"]
        )

        obj_to_ra = {
            str(location_table["object_id"][index]): location_table["ra"][index]
            for index in range(len(location_table))
        }
        obj_to_dec = {
            str(location_table["object_id"][index]): location_table["dec"][index]
            for index in range(len(location_table))
        }

        del location_table

        logger.info("Assembling Manifest...")

        # These are the column names expected in a manifest file by the downloader
        column_names = Downloader.MANIFEST_COLUMN_NAMES
        columns = {column_name: [] for column_name in column_names}

        # These will vary every object and must be implemented below
        dynamic_column_names = ["object_id", "filter", "dim", "tract", "ra", "dec", "filename"]
        # These are pulled from config ("sw", "sh", "rerun", "type", "image", "mask", and "variance")
        static_column_names = [name for name in column_names if name not in dynamic_column_names]

        # Check that all column names we need for a manifest are either in static or dynamic columns
        for column_name in column_names:
            if column_name not in static_column_names and column_name not in dynamic_column_names:
                raise RuntimeError(f"Error Assembling manifest {column_name} not implemented")

        static_values = {
            "sw": parse_degree(config["download"]["sw"]),
            "sh": parse_degree(config["download"]["sh"]),
            "rerun": parse_rerun(config["download"]["rerun"]),
            "type": parse_type(config["download"]["type"]),
            "image": parse_bool(config["download"]["image"]),
            "mask": parse_bool(config["download"]["mask"]),
            "variance": parse_bool(config["download"]["variance"]),
        }

        for index, (object_id, filter, filename, dim) in enumerate(self._all_files_full()):
            for static_col in static_column_names:
                columns[static_col].append(static_values[static_col])

            for dynamic_col in dynamic_column_names:
                if dynamic_col == "object_id":
                    columns[dynamic_col].append(int(object_id))
                elif dynamic_col == "filter":
                    columns[dynamic_col].append(filter)
                elif dynamic_col == "dim":
                    columns[dynamic_col].append(dim)
                elif dynamic_col == "tract":
                    # There's value in pulling tract from the filename rather than the download catalog
                    # in case The catalog had it wrong, the filename will have the value the cutout server
                    # provided.
                    tract = filename.split("_")[4]
                    columns[dynamic_col].append(parse_tract_opt(tract))
                elif dynamic_col == "ra":
                    ra = obj_to_ra[object_id]
                    columns[dynamic_col].append(parse_longitude(ra))
                elif dynamic_col == "dec":
                    dec = obj_to_dec[object_id]
                    columns[dynamic_col].append(parse_latitude(dec))
                elif dynamic_col == "filename":
                    columns[dynamic_col].append(filename)
                else:
                    # The tower of if statements has been entirely to create this failure path.
                    # which will be hit when someone alters dynamic column names above without also
                    # writing an implementation.
                    raise RuntimeError(f"No implementation to process column {dynamic_col}")
            if index != 0 and index % 1_000_000 == 0:
                logger.info(f"Addeed {index} objects to manifest")
        else:
            logger.info(f"Addeed {index + 1} objects to manifest")

        logger.info("Writing rebuilt manifest...")
        manifest_table = Table(columns)

        manifest_file_path = Path(config["general"]["data_dir"]) / Downloader.MANIFEST_FILE_NAME

        # Rename the old manifest
        if manifest_file_path.exists():
            filename_safe_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            new_file_name = Downloader.MANIFEST_FILE_NAME + f".archived.at.{filename_safe_now}"
            manifest_file_path.rename(Path(config["general"]["data_dir"]) / new_file_name)

        # Replace the old manifest
        manifest_table.write(manifest_file_path, overwrite=True, format="fits")

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

    def __getitem__(self, idx: int) -> Tensor:
        if idx >= len(self.files) or idx < 0:
            raise IndexError

        # Use the list of object IDs for explicit indexing
        object_id = list(self.files.keys())[idx]

        return self._object_id_to_tensor(object_id)

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
        return object_id in list(self.files.keys()) and object_id in list(self.dims.keys())

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
                logger.info(f"Processed {index} objects")
            yield str(object_id)
        else:
            if log:
                logger.info(f"Processed {index} objects")

    def _all_files_full(self):
        """
        Private read-only iterator over all files that enforces a strict total order across
        objects and filters. Will not work prior to self.files, and self.path initialization in __init__

        Yields
        ------
        Tuple[object_id, filter, filename, dim]
            Members of this tuple are
            - The object_id as a string
            - The filter name as a string
            - The filename relative to self.path
            - A tuple containing the dimensions of the fits file in pixels.
        """
        for object_id in self.ids():
            dims = self.dims[object_id]
            for idx, (filter, filename) in enumerate(self._filter_filename(object_id)):
                yield (object_id, filter, filename, dims[idx])

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

    def _object_files_filters_ref(self, object_id):
        """
        Private read-only iterator over all files for a given object. This enforces a strict total order
        across filters. Will not work prior to self.files, and self.path initialization in __init__

        Guaranteed to only return files that have filters in self.filters_ref.

        Yields
        ------
        Path
            The path to the file.
        """
        for filter, filename in self._filter_filename(object_id):
            if filter in self.filters_ref:
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

        logger.info("Preloading HSCDataSet cache...")

        with ThreadPoolExecutor(max_workers=HSCDataSet._determine_numprocs_preload()) as executor:
            tensors = self._lazy_map_executor(executor, self.ids(log_every=1_000_000))

            start_time = time.monotonic_ns()
            for idx, (id, tensor) in enumerate(zip(self.ids(), tensors)):
                self.tensors[id] = tensor

                # Output timing every 1k tensors
                if idx % 1_000 == 0 and idx != 0:
                    self._log_duration_tensorboard("HSCDataSet/preload_1k_obj_s", start_time)
                    start_time = time.monotonic_ns()

    def _lazy_map_executor(self, executor: Executor, ids: Iterable[str]) -> Iterator[Tensor]:
        """This is a version of concurrent.futures.Executor map() which lazily evaluates the iterator passed
        We do this because we do not want all of the tensors to remain in memory during pre-loading. We would
        prefer a smaller set of in-flight tensors.

        The total number of in progress jobs is set at HSCDataSet._determine_numprocs().

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

        max_futures = HSCDataSet._determine_numprocs_preload()
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
        if self.tensorboardx_logger:
            since_tensorboard_start_us = (start_time - self.tensorboard_start_ns) / 1.0e3

            duration_s = (now - start_time) / 1.0e9
            self.tensorboardx_logger.add_scalar(name, duration_s, since_tensorboard_start_us)

    def _check_object_id_to_tensor_cache(self, object_id: str) -> Optional[Tensor]:
        return self.tensors.get(object_id, None)

    def _populate_object_id_to_tensor_cache(self, object_id: str) -> Tensor:
        data_torch = self._read_object_id(object_id)
        self.tensors[object_id] = data_torch
        return data_torch

    def _read_object_id(self, object_id: str) -> Tensor:
        start_time = time.monotonic_ns()

        # Read all the files corresponding to this object
        data = []

        for filepath in self._object_files_filters_ref(object_id):
            file_start_time = time.monotonic_ns()
            raw_data = fits.getdata(filepath, memmap=False)
            data.append(raw_data)
            self._log_duration_tensorboard("HSCDataSet/file_read_time_s", file_start_time)

        self._log_duration_tensorboard("HSCDataSet/object_read_time_s", start_time)

        data_torch = self._convert_to_torch(data)
        self._log_duration_tensorboard("HSCDataSet/object_total_read_time_s", start_time)
        return data_torch

    def _convert_to_torch(self, data: list[npt.ArrayLike]) -> Tensor:
        start_time = time.monotonic_ns()

        # Push all the filter data into a tensor object
        data_np = np.array(data)
        data_torch = from_numpy(data_np.astype(np.float32))

        # Apply our transform stack
        data_torch = self.transform(data_torch) if self.transform is not None else data_torch

        self._log_duration_tensorboard("HSCDataSet/object_convert_tensor_time_s", start_time)
        return data_torch

    # TODO: Performance Change when files are read/cache pytorch tensors?
    #
    # This function loads from a file every time __getitem__ is called
    # Do we want to pre-cache these into memory in init?
    # Do we want to memoize them on first __getitem__ call?
    #
    # For now we just do it the naive way
    def _object_id_to_tensor(self, object_id: str) -> Tensor:
        """Converts an object_id to a pytorch tensor with dimenstions (self.num_filters, self.cutout_shape[0],
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
            self._log_duration_tensorboard("HSCDataSet/cache_hit_s", start_time)
            return data_torch

        data_torch = self._populate_object_id_to_tensor_cache(object_id)
        self._log_duration_tensorboard("HSCDataSet/cache_miss_s", start_time)
        return data_torch

    def _prepare_metadata(self) -> Optional[Table]:
        if self.filter_catalog_table is None:
            return None

        # This happens when filter_catalog_table is injected in unit tests
        if isinstance(self.filter_catalog_table, str):
            return None

        # We're going to be operating on object_id and filter columns en masse.
        self.filter_catalog_table.add_index(["object_id", "filter"])

        # Get all object_ids in enumeration order
        sorted_object_ids = np.array([int(id) for id in self.ids()])

        # Get a single row per object_id by selecting for a single filter
        filter_catalog_table_dedup = self.filter_catalog_table.loc["filter", self.filters_ref[0]]

        # Get all rows in enumeration order using the object_ids
        metadata = filter_catalog_table_dedup.loc["object_id", sorted_object_ids]

        # Filter for the appropriate columns
        colnames = list(self.filter_catalog_table.colnames)
        colnames.remove("filename")
        colnames.remove("filter")

        return metadata[colnames]
