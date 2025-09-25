import logging
from collections.abc import Generator
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from hyrax.config_utils import find_most_recent_results_dir

from .data_set_registry import HyraxDataset

logger = logging.getLogger(__name__)

ORIGINAL_DATASET_CONFIG_FILENAME = "original_dataset_config.toml"


class InferenceDataSet(HyraxDataset, Dataset):
    """This is a dataset class to represent the situations where we wish to treat the output of inference
    as a dataset. e.g. when performing umap/visualization operations"""

    def __init__(
        self,
        config,
        results_dir: Optional[Union[Path, str]] = None,
        verb: Optional[str] = None,
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

        # Open the batch index numpy file.
        # Loop over files and create if it does not exist
        batch_index_path = self.results_dir / "batch_index.npy"
        if not batch_index_path.exists():
            msg = f"{self.results_dir} is corrupt and lacks a batch index file."
            raise RuntimeError(msg)

        self.batch_index = np.load(self.results_dir / "batch_index.npy")
        self.length = len(self.batch_index)

        # Initializes our first element. This primes the cache for sequential access
        # as well as giving us a sample element for shape()
        self.cached_batch_num: Optional[int] = None

        self.shape_element = self._load_from_batch_file(
            self.batch_index["batch_num"][0], self.batch_index["id"][0]
        )[0]

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
        ConfigDict
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

        # Ensure the cached batch is loaded
        if self.cached_batch_num is None or batch_num != self.cached_batch_num:
            self.cached_batch_num = batch_num
            self.cached_batch: np.ndarray = np.load(self.results_dir / f"batch_{batch_num}.npy")

        return self.cached_batch[np.isin(self.cached_batch["id"], ids)]

    def _resolve_results_dir(
        self, config, results_dir: Optional[Union[Path, str]], verb: Optional[str]
    ) -> Path:
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
    """Class to write out inference datasets. Used by infer, umap to consistently write out numpy
    files in batches which can be read by InferenceDataSet.

    With the exception of building ID->Batch indexing info, this is implemented as a bag-o-functions that
    manipulate the filesystem directly as their primary effect.
    """

    def __init__(self, original_dataset: Dataset, result_dir: Union[str, Path]):
        """
        .. py:method:: __init__

        """
        self.result_dir = result_dir if isinstance(result_dir, Path) else Path(result_dir)
        self.batch_index = 0

        # Detect the dtype numpy will want to use for ids for the original dataset
        self.id_dtype = np.array(list(original_dataset.ids())).dtype

        self.all_ids = np.array([], dtype=self.id_dtype)
        self.all_batch_nums = np.array([], dtype=np.int64)
        self.writer_pool = Pool()

        # If we're being asked to write an InferenceDataset based on another InferenceDataset then we
        # Use the backing InferenceDataset's original config, which is presumably a non-InferenceDataset
        # Otherwise just use the config of the dataset given.
        self.original_dataset_config = (
            original_dataset.original_config
            if hasattr(original_dataset, "original_config")
            else original_dataset.config
        )

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
