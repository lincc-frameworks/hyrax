import logging
from collections.abc import Generator
from multiprocessing import get_context
from pathlib import Path
from typing import Union

import lancedb
import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from hyrax.config_utils import find_most_recent_results_dir

from .data_set_registry import HyraxDataset

logger = logging.getLogger(__name__)

# Storage backend constants
STORAGE_NPY = "npy"
STORAGE_LANCE = "lance"
STORAGE_AUTO = "auto"

ORIGINAL_DATASET_CONFIG_FILENAME = "original_dataset_config.toml"


class InferenceDataSet(HyraxDataset, Dataset):
    """This is a dataset class to represent the situations where we wish to treat the output of inference
    as a dataset. e.g. when performing umap/visualization operations"""

    def __init__(
        self,
        config,
        results_dir: Union[Path, str] | None = None,
        verb: str | None = None,
    ):
        """Initialize an InferenceDataSet object.

        As a user of this code, you should almost never create this class, Instances of this class are
        returned by the umap and infer verbs. Prefer those over creating your own.

        If you do end up creating your own class, you will need a hyrax config, and to know some things
        about where the result you are interested in is stored.

        Parameters
        ----------
        config : dict
            The hyrax config dictionary
        results_dir : Optional[Union[Path, str]], optional
            The results subdirectory of the inference or umap results you want to access, by default None.
            If no results subdirectory is provided, this function will attempt the following in order:

            #. Use the directory specified in ``config['results']['inference_dir']`` if set and the directory
               exists
            #. Look in the results configured in ``config['general']['results_dir']`` (``./results/``
               by default), then use the most recent results directory corresponding to the verb specified.

        verb : Optional[str], optional
            The name of the verb that generated the results, only important when the most recent results
            are being fetched. If no verb is provided, "infer" will be assumed.

        Raises
        ------
        RuntimeError
            When the provided results directory is corrupt, or cannot be found.
        """
        from hyrax.config_utils import ConfigManager
        from hyrax.pytorch_ignite import setup_dataset

        super().__init__(config)
        self.results_dir = self._resolve_results_dir(config, results_dir, verb)

        # Detect storage format and initialize appropriate backend
        self.storage_format = self._detect_storage_format()
        logger.debug(f"Detected storage format: {self.storage_format}")

        if self.storage_format == STORAGE_LANCE:
            self._init_lance_backend()
        else:
            self._init_npy_backend()

        # Initializes our first element. This primes the cache for sequential access
        # as well as giving us a sample element for shape()
        self.shape_element = self._load_element_sample()

        # Initialize the original dataset using the old config, so we have it
        # around for metadata calls
        self._original_dataset_config = ConfigManager().read_runtime_config(
            self.results_dir / ORIGINAL_DATASET_CONFIG_FILENAME
        )

        # Disable cache preloading on this dataset because it will only be used for its metadata
        # TODO: May want to add some sort of metadata_only optional arg to dataset constructor
        #       so we can opt-out of expensive dataset operations conditional on us only needing metadata
        #
        #       Alternatively this may be an opportunity for a metadata mixin sort of class structure where
        #       we can bring up Only the metadata for a dataset, without constructing the whole thing.
        self._original_dataset_config["data_set"]["preload_cache"] = False
        self.original_dataset = setup_dataset(self._original_dataset_config)  # type: ignore[arg-type]
        self.original_dataset = (
            self.original_dataset["infer"]
            if isinstance(self.original_dataset, dict)
            else self.original_dataset
        )

    def _detect_storage_format(self) -> str:
        """Detect which storage format is used in the results directory.

        Returns
        -------
        str
            Either STORAGE_LANCE or STORAGE_NPY
        """
        lance_path = self.results_dir / "inference_data.lance"
        npy_path = self.results_dir / "batch_index.npy"

        if lance_path.exists():
            return STORAGE_LANCE
        elif npy_path.exists():
            # Check if auto-migration is enabled
            if self.config["general"].get("storage_backend") == STORAGE_AUTO and self.config["general"].get(
                "auto_migrate_to_lance", False
            ):
                logger.info(f"Auto-migrating .npy files to LanceDB format in {self.results_dir}")
                self._migrate_npy_to_lance()
                return STORAGE_LANCE
            return STORAGE_NPY
        else:
            msg = f"{self.results_dir} is corrupt and lacks both Lance and .npy data files."
            raise RuntimeError(msg)

    def _init_lance_backend(self):
        """Initialize the LanceDB backend for reading inference results."""

        db = lancedb.connect(str(self.results_dir))
        self.lance_table = db.open_table("inference_data")

        # Get length and create index mapping
        self.length = self.lance_table.count_rows()

        # Cache the batch index as a numpy array for compatibility
        # We'll load it lazily when needed
        self._lance_batch_index = None

        # Detect tensor dtype and shape from first record
        # Use head() to get PyArrow table, then convert to numpy
        first_batch = self.lance_table.head(1)
        first_row = first_batch.to_pydict()
        # tensor is stored as nested list, convert to numpy to get dtype/shape
        sample_tensor = np.array(first_row["tensor"][0])
        self.tensor_dtype = sample_tensor.dtype
        self.tensor_shape = sample_tensor.shape

    def _init_npy_backend(self):
        """Initialize the numpy backend for reading inference results."""
        batch_index_path = self.results_dir / "batch_index.npy"
        if not batch_index_path.exists():
            msg = f"{self.results_dir} is corrupt and lacks a batch index file."
            raise RuntimeError(msg)

        self.batch_index = np.load(batch_index_path)
        self.length = len(self.batch_index)
        self.cached_batch_num: int | None = None

    def _load_element_sample(self):
        """Load a sample element for shape detection."""
        if self.storage_format == STORAGE_LANCE:
            # Get first row from Lance table as PyArrow, convert to numpy properly
            first_batch = self.lance_table.head(1)
            first_row = first_batch.to_pydict()
            # Convert nested list to numpy array with proper dtype
            tensor_data = np.array(first_row["tensor"][0], dtype=self.tensor_dtype).reshape(self.tensor_shape)
            # Create structured array with proper dtype
            dtype = [("id", object), ("tensor", self.tensor_dtype, self.tensor_shape)]
            result = np.zeros(1, dtype=dtype)
            result[0]["id"] = first_row["id"][0]
            result[0]["tensor"][:] = tensor_data
            return result[0]
        else:
            # Original .npy behavior
            return self._load_from_batch_file(self.batch_index["batch_num"][0], self.batch_index["id"][0])[0]

    def _migrate_npy_to_lance(self):
        """Migrate existing .npy files to LanceDB format."""
        # Load the batch index
        batch_index_path = self.results_dir / "batch_index.npy"
        if not batch_index_path.exists():
            msg = f"Cannot migrate: {self.results_dir} lacks batch_index.npy"
            raise RuntimeError(msg)

        batch_index = np.load(batch_index_path)

        # Collect all data from batch files
        all_data = []
        unique_batches = np.unique(batch_index["batch_num"])

        logger.info(f"Migrating {len(unique_batches)} batch files to LanceDB...")

        for batch_num in unique_batches:
            batch_file = self.results_dir / f"batch_{batch_num}.npy"
            if batch_file.exists():
                batch_data = np.load(batch_file)
                for record in batch_data:
                    all_data.append(
                        {
                            "id": str(record["id"]),
                            "batch_num": int(batch_num),
                            "tensor": record["tensor"].tolist(),
                        }
                    )

        # Create Lance table - PyArrow will handle nested list schema automatically
        db = lancedb.connect(str(self.results_dir))
        db.create_table("inference_data", data=all_data, mode="overwrite")

        logger.info(f"Successfully migrated {len(all_data)} records to LanceDB format")

    @property
    def batch_index(self):
        """Get batch index, loading from Lance if necessary."""
        if self.storage_format == STORAGE_LANCE:
            if self._lance_batch_index is None:
                # Convert Lance table to batch_index format for compatibility
                # Use PyArrow table for better performance
                pa_table = self.lance_table.to_arrow()
                data_dict = pa_table.to_pydict()
                dtype = np.dtype([("id", object), ("batch_num", np.int64)])
                self._lance_batch_index = np.zeros(len(data_dict["id"]), dtype=dtype)
                self._lance_batch_index["id"] = np.array(data_dict["id"], dtype=object)
                self._lance_batch_index["batch_num"] = np.array(data_dict["batch_num"], dtype=np.int64)
            return self._lance_batch_index
        else:
            return self._batch_index

    @batch_index.setter
    def batch_index(self, value):
        """Set batch index (only used for .npy backend)."""
        if self.storage_format == STORAGE_NPY:
            self._batch_index = value

    def _shape(self):
        """The shape of the dataset (Discovered from files)

        Returns
        -------
        Tuple
            Tuple with the shape of an individual element of the dataset
        """
        # Note: our __getitem__() needs self.shape() to work. We cannot use HraxDataset.shape()
        # because that shape uses __getitem__(), so we must define this for ourselves
        return self.shape_element["tensor"].shape

    def ids(self) -> Generator[str]:
        """IDs of this dataset. Will return a string generator with IDs.

        These IDs are the IDs of the dataset used originally to generate this dataset.

        Returns
        -------
        Generator[str]
            Generator that yields the string ids of this dataset

        Yields
        ------
        Generator[str]
            Yields the string ids of this dataset
        """
        # Note: Not using HyraxDataset.ids() here because we need to return the ids of whatever
        # dataset was used for inference, not the sequential index HyraxDataset.ids() gives.
        return (str(id) for id in self.batch_index["id"])

    def __getitem__(self, idx: Union[int, np.ndarray]):
        """Implements the ``[]`` operator

        Parameters
        ----------
        idx : Union[int, np.ndarray]
            Either an index or a numpy array of indexes.
            These are NOT the ID values of the dataset, but rather a zero-based index starting
            at the beginning of the inference dataset.

        Returns
        -------
        torch.tensor
            Either the tensor corresponding to a single result, or a tensor with a multiplicity of
            results if multiple indexes were passed.
        """
        from torch import from_numpy

        try:
            _ = (e for e in idx)  # type: ignore[union-attr]
        except TypeError:
            idx = np.array([idx])

        # Allocate a numpy array to hold all the tensors we will get in order
        # Needs to be the appropriate shape
        shape_tuple = tuple([len(idx)] + list(self._shape()))
        all_tensors = np.zeros(shape=shape_tuple)

        # We need to look up all the batches for the ids we get
        lookup_batch = self.batch_index[idx]

        # We then need to sort the resultant id->batch catalog by batch
        original_indexes = np.argsort(lookup_batch, order="batch_num")
        sorted_lookup_batches = np.take_along_axis(lookup_batch, original_indexes, axis=-1)

        unique_batch_nums = np.unique(sorted_lookup_batches["batch_num"])
        for batch_num in unique_batch_nums:
            # Mask our batch out to get IDs and the original indexes it had in the query
            batch_mask = sorted_lookup_batches["batch_num"] == batch_num
            batch_ids = sorted_lookup_batches[batch_mask]["id"]
            batch_original_indexes = original_indexes[batch_mask]

            # Lookup in each batch file
            batch_tensors = np.sort(self._load_from_batch_file(batch_num, batch_ids), order="id")

            # Place the resulting tensors in the results array where they go.
            all_tensors[batch_original_indexes] = batch_tensors["tensor"]

        # In the case of a single id this will be a tensor that has the appropriate shape
        # Otherwise we will have a stacked array of tensors
        all_tensors = all_tensors[0] if len(all_tensors) == 1 else all_tensors

        return from_numpy(all_tensors)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return self.length

    @property
    def original_config(self) -> dict:
        """Get the original configuration for the dataset used to generate this inference dataset

        Since this sort of dataset is definitionally an intermediate product, this returns the
        runtime config used to construct that dataset rather than this one.

        Returns
        -------
        dict
            Configuration that can be used to create the original dataset that was used
            as input for whatever inference process created this dataset.
        """
        return self._original_dataset_config

    def metadata_fields(self) -> list[str]:
        """Get the metadata fields associted with the original dataset used to generate this one

        Returns
        -------
        list[str]
            List of valid field names for metadata queries
        """
        # We must override this and pass to the original dataset.
        return self.original_dataset.metadata_fields()  # type: ignore[no-any-return,attr-defined]

    def metadata(self, idxs: npt.ArrayLike, fields: list[str]) -> npt.ArrayLike:
        """Get metadata associated with the data in the InferenceDataSet. This metadata comes from
        the original dataset, but is indexed according to the InferenceDataSet.

        Parameters
        ----------
        idxs : npt.ArrayLike
            Indexes in the InferenceDataSet for which metadata is desired
        fields : list[str]
            Metadata fields requested

        Returns
        -------
        npt.ArrayLike
            An array where the rows correspond to the passed list of indexes and the columns
            correspond to the fields passed. Order is preserved- metadata[i] corresponds to idxs[i].
        """

        idxs = np.asarray(idxs)

        # Get the requested IDs in the order they were requested
        ids_requested = np.array(list(self.ids()))[idxs]  # type: ignore[index]

        # Get all original dataset IDs
        original_ids = np.array(list(self.original_dataset.ids()))  # type: ignore[attr-defined]

        # Create mapping from original ID to original index
        id_to_original_idx = {str(oid): i for i, oid in enumerate(original_ids)}

        # Map requested IDs to original indices, preserving order
        original_idxs = [id_to_original_idx[str(req_id)] for req_id in ids_requested]

        # Get metadata from original dataset
        original_metadata = self.original_dataset.metadata(original_idxs, fields)  # type: ignore[attr-defined,no-any-return]

        # Return metadata in the same order as requested
        return original_metadata

    def _load_from_batch_file(self, batch_num: int, ids=Union[int, np.ndarray]) -> np.ndarray:
        """Hands back an array of tensors given a set of IDs in a particular batch and the given
        batch number"""

        if self.storage_format == STORAGE_LANCE:
            # Query Lance table for specific batch and IDs
            # Ensure ids is iterable
            if not isinstance(ids, np.ndarray):
                ids = np.array([ids])

            # Convert ids to strings for querying
            id_strs = [str(id_val) for id_val in ids]

            # Query the table using Lance SQL-like syntax
            # Get full table as PyArrow then filter in memory (more efficient for small queries)
            pa_table = self.lance_table.to_arrow()
            data_dict = pa_table.to_pydict()

            # Filter by batch_num and id
            filtered_indices = [
                i
                for i, (bid, bnum) in enumerate(zip(data_dict["id"], data_dict["batch_num"]))
                if bnum == batch_num and bid in id_strs
            ]

            # Use the same dtype as shape_element to ensure compatibility
            result = np.zeros(len(filtered_indices), dtype=self.shape_element.dtype)
            for result_idx, table_idx in enumerate(filtered_indices):
                result[result_idx]["id"] = data_dict["id"][table_idx]
                # Convert nested list to numpy array
                tensor_list = data_dict["tensor"][table_idx]
                tensor_data = np.array(tensor_list, dtype=self.tensor_dtype).reshape(self.tensor_shape)
                # Assign using slice notation to copy into the structured array
                result[result_idx]["tensor"][:] = tensor_data

            return result
        else:
            # Original .npy behavior
            # Ensure the cached batch is loaded
            if self.cached_batch_num is None or batch_num != self.cached_batch_num:
                self.cached_batch_num = batch_num
                self.cached_batch: np.ndarray = np.load(self.results_dir / f"batch_{batch_num}.npy")

            return self.cached_batch[np.isin(self.cached_batch["id"], ids)]

    def _resolve_results_dir(self, config, results_dir: Union[Path, str] | None, verb: str | None) -> Path:
        """Initialize an inference results directory as a data source. Accepts an override of what
        directory to use"""

        verb = "infer" if verb is None else verb

        if results_dir is None:
            if self.config["results"]["inference_dir"]:
                results_dir = self.config["results"]["inference_dir"]
                if not isinstance(results_dir, str):
                    msg = "Configured [results_dir] is not a string"
                    raise RuntimeError(msg)
            else:
                results_dir = find_most_recent_results_dir(self.config, verb=verb)
                if results_dir is None:
                    msg = "Could not find a results directory. Run infer or use "
                    msg += "[results] inference_dir config to specify a directory."
                    raise RuntimeError(msg)
                msg = f"Using most recent results dir {results_dir} for lookup."
                msg += " Use the [results] inference_dir config to set a directory or pass it to this verb."
                logger.debug(msg)

        retval = Path(results_dir) if isinstance(results_dir, str) else results_dir

        if not retval.exists():
            msg = f"Inference directory {results_dir} does not exist"
            raise RuntimeError(msg)

        return retval


class InferenceDataSetWriter:
    """Class to write out inference datasets. Supports both .npy and LanceDB formats.

    With the exception of building ID->Batch indexing info, this is implemented as a bag-o-functions that
    manipulate the filesystem directly as their primary effect.
    """

    def __init__(self, original_dataset: Dataset, result_dir: Union[str, Path], config: dict | None = None):
        """
        .. py:method:: __init__

        Parameters
        ----------
        original_dataset : Dataset
            The dataset being processed
        result_dir : Union[str, Path]
            Directory where results will be written
        config : dict, optional
            Hyrax configuration dictionary. If None, will attempt to get from original_dataset
        """
        self.result_dir = result_dir if isinstance(result_dir, Path) else Path(result_dir)
        self.batch_index = 0

        # Detect the dtype numpy will want to use for ids for the original dataset
        self.id_dtype = np.array(list(original_dataset.ids())).dtype

        self.all_ids = np.array([], dtype=self.id_dtype)
        self.all_batch_nums = np.array([], dtype=np.int64)

        # Determine storage backend
        if config is None:
            config = getattr(original_dataset, "config", {"general": {"storage_backend": STORAGE_AUTO}})
        self.config = config
        self.storage_backend = config.get("general", {}).get("storage_backend", STORAGE_AUTO)

        # For "auto" mode, default to Lance
        if self.storage_backend == STORAGE_AUTO:
            self.storage_backend = STORAGE_LANCE
            logger.info("Using LanceDB storage format (convertible to Pandas/Parquet)")

        # Initialize storage-specific components
        if self.storage_backend == STORAGE_LANCE:
            self._init_lance_writer()
        else:
            self._init_npy_writer()

        # If we're being asked to write an InferenceDataset based on another InferenceDataset then we
        # Use the backing InferenceDataset's original config, which is presumably a non-InferenceDataset
        # Otherwise just use the config of the dataset given.
        self.original_dataset_config = (
            original_dataset.original_config
            if hasattr(original_dataset, "original_config")
            else original_dataset.config
        )

    def _init_lance_writer(self):
        """Initialize LanceDB writer."""
        self.lance_data = []
        self.lance_db = None  # Will be created in write_index

    def _init_npy_writer(self):
        """Initialize numpy writer."""
        # Create a multiprocessing pool to write batches in parallel. We specifically
        # use the "fork" context because Hyrax makes heavy use of delayed imports
        # which will cause crashes if we use "spawn" (the default on Windows and MacOS)
        self.writer_pool = get_context("fork").Pool()

    def write_batch(self, ids: np.ndarray, tensors: list[np.ndarray]):
        """Write a batch of tensors into the dataset. This writes the whole batch immediately.
        Caller is in charge of batch size consistency considerations, and that ids is the same length as
        tensors

        Parameters
        ----------
        ids : np.ndarray
            Array of IDs, dtype of the elements must match the dtype type of the ids of the original dataset
            used to construct this InferenceDataSetWriter.
        tensors : list[np.ndarray]
            List of consistently dimensioned numpy arrays to save.
        """
        batch_len = len(tensors)

        if self.storage_backend == STORAGE_LANCE:
            # Add data to Lance buffer
            # Note: PyArrow can't handle multi-dimensional arrays directly,
            # so we convert to lists. Shape/dtype info is stored separately.
            for i, tensor in enumerate(tensors):
                self.lance_data.append(
                    {"id": str(ids[i]), "batch_num": int(self.batch_index), "tensor": tensor.tolist()}
                )
        else:
            # Original .npy behavior
            # Save results from this batch in a numpy file as a structured array
            first_tensor = tensors[0]
            structured_batch_type = np.dtype(
                [("id", self.id_dtype), ("tensor", first_tensor.dtype, first_tensor.shape)]
            )
            structured_batch = np.zeros(batch_len, structured_batch_type)
            structured_batch["id"] = ids
            structured_batch["tensor"] = tensors

            filename = f"batch_{self.batch_index}.npy"
            savepath = self.result_dir / filename
            if savepath.exists():
                RuntimeError(f"Writing objects in batch {self.batch_index} but {filename} already exists.")

            self.writer_pool.apply_async(
                func=np.save, args=(savepath, structured_batch), kwds={"allow_pickle": False}
            )

        self.all_ids = np.append(self.all_ids, ids)
        self.all_batch_nums = np.append(self.all_batch_nums, np.full(batch_len, self.batch_index))

        self.batch_index += 1

    def write_index(self):
        """Writes out the batch index built up by this object over multiple write_batch calls.
        See save_batch_index for details.
        """
        from hyrax.config_utils import log_runtime_config

        if self.storage_backend == STORAGE_LANCE:
            # Write all data to Lance table
            if self.lance_data:
                db = lancedb.connect(str(self.result_dir))
                # LanceDB will automatically infer schema from the data
                # PyArrow will preserve nested list structure
                db.create_table("inference_data", data=self.lance_data, mode="overwrite")
                logger.debug(f"Wrote {len(self.lance_data)} records to LanceDB table")
        else:
            # Original .npy behavior
            # First ensure we are done writing out all batches
            self.writer_pool.close()
            self.writer_pool.join()

            # Then write out the batch index.
            self._save_batch_index()

        # Write out the config needed to re-constitute the original dataset we came from.
        log_runtime_config(self.original_dataset_config, self.result_dir, ORIGINAL_DATASET_CONFIG_FILENAME)

    def _save_batch_index(self):
        """Save a batch index in the result directory provided"""
        batch_index_dtype = np.dtype([("id", self.id_dtype), ("batch_num", np.int64)])
        batch_index = np.zeros(len(self.all_ids), batch_index_dtype)
        batch_index["id"] = np.array(self.all_ids)
        batch_index["batch_num"] = np.array(self.all_batch_nums)

        # Save the batch index in insertion order
        filename = "batch_index_insertion_order.npy"
        self._save_file(filename, batch_index)

        # Sort the batch index by id, and save it again
        batch_index.sort(order="id")
        filename = "batch_index.npy"
        self._save_file(filename, batch_index)

    def _save_file(self, filename: str, data: np.ndarray):
        """Save a numpy array to a file in the result directory provided"""
        savepath = self.result_dir / filename
        if savepath.exists():
            raise RuntimeError(f"The path to save {filename} already exists.")
        np.save(savepath, data, allow_pickle=False)
