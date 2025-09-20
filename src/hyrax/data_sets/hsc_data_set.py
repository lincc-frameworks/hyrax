# ruff: noqa: D101, D102

import datetime
import logging
import multiprocessing
import os
import re
import resource
from pathlib import Path
from typing import Optional

import numpy as np
from schwimmbad import MultiPool

from hyrax.config_utils import ConfigDict

from .fits_image_dataset import FitsImageDataSet, files_dict

logger = logging.getLogger(__name__)
dim_dict = dict[str, list[tuple[int, int]]]


class HSCDataSet(FitsImageDataSet):
    """Dataset for sets of HSC cutouts created by the ``fibad download`` command."""

    _called_from_test = False

    def __init__(self, config: ConfigDict, data_location=None):
        """
        .. py:method:: __init__

        """
        from hyrax.download import Downloader

        # Note "rebuild_manifest" is not a config, its a hack for rebuild_manifest mode
        # to ensure we don't use the manifest we believe is corrupt.
        rebuild_manifest = config["rebuild_manifest"] if "rebuild_manifest" in config else False  # noqa: SIM401

        # Set the filter catalog
        # If we are in rebuild manifest mode don't use any filter catalog
        if rebuild_manifest:
            config["data_set"]["filter_catalog"] = False
        # If there's no filter catalog, try to use the manifest file if it exists
        elif not config["data_set"]["filter_catalog"]:
            catalog = Path(data_location) / Downloader.MANIFEST_FILE_NAME
            if catalog.exists():
                config["data_set"]["filter_catalog"] = str(catalog.expanduser().resolve())

        self.filters_config = config["data_set"]["filters"] if config["data_set"]["filters"] else None

        super().__init__(config, data_location)

    def _read_filter_catalog(self, filter_catalog_path: Optional[Path]):
        from astropy.table import Table

        try:
            retval = super()._read_filter_catalog(filter_catalog_path)
        except RuntimeError:
            # _read_filter_catalog is persnickity about filter_catalog_path.
            # Ignore all of the error checking in there and _parse_filter_catalog
            # will try to recover if the table is malformed/missing.
            retval = None

        if isinstance(retval, Table):
            colnames = retval.colnames
            if (self.filter_column_name not in colnames) ^ (self.filename_column_name not in colnames):
                msg = f"Filter catalog file {filter_catalog_path} provides one of filters or filenames "
                msg += "without the other. Filesystem scan will still occur without both defined."
                logger.warning(msg)

        return retval

    # The main job of this function is to transmute the filter catalog table into
    # the dictionaries that the rest of the class uses.
    #
    # In the HSC case this will also have to do fallback and call
    # _scan_file_dimensions() and/or _scan_file_names() and pass back only the files dict.
    def _parse_filter_catalog(self, table) -> None:
        object_id_missing = self.object_id_column_name not in table.colnames if table is not None else True
        filter_missing = self.filter_column_name not in table.colnames if table is not None else True
        filename_missing = self.filename_column_name not in table.colnames if table is not None else True

        file_scan = table is None or object_id_missing or filter_missing or filename_missing

        object_ids_for_filescan = None
        if not object_id_missing:
            if filter_missing and filename_missing:
                object_ids_for_filescan = list(table[self.object_id_column_name])
            elif filter_missing or filename_missing:
                object_ids_for_filescan = list(set(table[self.object_id_column_name]))

        # Detect the list of filters, but allow config based override
        if file_scan:
            self.files = self._scan_file_names(self.filters_config, object_ids_for_filescan)
            self.dims = self._scan_file_dimensions()

        # Otherwise we have a well formed table
        else:
            # Have the superclass assemble self.files
            self.files = super()._parse_filter_catalog(table)

            # Assemble dims for ourself if the column is available or fallback to self._scan_file_dimensions()
            if "dim" not in table.colnames:
                self.dims = self._scan_file_dimensions()
            else:
                dim_catalog: dim_dict = {}

                for row in table:
                    object_id = str(row[self.object_id_column_name])
                    # filter = row[self.filter_column_name]
                    filename = row[self.filename_column_name]
                    dim = tuple(row["dim"])

                    # Skip over any files that are marked as didn't download or have <1x1 size, removing the
                    # relevant object from the files dict if it exists
                    if filename == "Attempted" or min(dim) < 1:
                        if object_id in self.files:
                            del self.files[object_id]
                        continue

                    # Dimension is optional, insert into dimension catalog.
                    if object_id not in dim_catalog:
                        dim_catalog[object_id] = []
                    dim_catalog[object_id].append(dim)

                self.dims = dim_catalog

        return self.files

    def _set_crop_transform(self):
        cutout_shape = self.config["data_set"]["crop_to"] if self.config["data_set"]["crop_to"] else None
        self.cutout_shape = self._check_file_dimensions() if cutout_shape is None else cutout_shape
        self.set_crop_transform(self.cutout_shape)

    def _before_preload(self):
        self.filters_ref = (
            list(list(self.files.values())[0]) if self.filters_config is None else self.filters_config
        )
        self.pruned_objects: dict[str, str] = {}
        self._prune_objects(self.filters_ref, self.cutout_shape)

    def _scan_file_names(
        self, filters: Optional[list[str]], filter_obj_ids: Optional[list[str]] = None
    ) -> files_dict:
        """Class initialization helper

        Parameters
        ----------
        filters: list[str], Optional:
            List of filters that we should look for in the data corpus

        filter_obj_ids: list[str], Optional:
            Filter the file scan to only file names which have the provided object IDs, skipping other files
            When not provided, all file names in the configured data directory that match the pattern from
            hyrax download are parsed.

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
            if isinstance(filter_obj_ids, list) and filename[:17] not in filter_obj_ids:
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

        # So we can use super() with no args inside the generator expression below
        super_obj = super()
        retval = {}
        with MultiPool(processes=HSCDataSet._determine_numprocs()) as pool:
            args = (
                (object_id, list(super_obj._object_files(object_id)))
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
        from astropy.io import fits

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

        cutout_shape: tuple[int, int]
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
        from astropy.table import Table

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

    def _object_files(self, object_id):
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
