import json
import logging
from collections.abc import Generator, Iterable
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from hyrax.config_utils import find_most_recent_results_dir, log_runtime_config

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
        self.results_dir = self._resolve_results_dir(results_dir, verb)
        logger.warning(f"Found results directory: {self.results_dir}")

        parquet_output = self.results_dir / "output.parquet"
        self.parquet_output = pq.read_table(parquet_output)
        self.data = self.parquet_output.to_pandas()
        self.model_output_shape = json.loads(self.parquet_output.schema.metadata.get(b"model_output_shape"))

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
        return (str(id) for id in self.parquet_output["id"])

    def get_model_output(self, idx):
        """Retrieve tensors from the pandas dataframe and reshape them to their original
        dimensions using the model_output_shape metadata.

        Parameters
        ----------
        idx : Union[int, slice, Iterable]
            The index or indicies of the tensor to retrieve.

        Returns
        -------
        np.array
            The tensor(s) corresponding to the input index(es) reshaped to the
            original dimensions.
        """
        model_output = self.data.iloc[idx]["model_output"]
        return np.array([o.reshape(self.model_output_shape) for o in model_output])

    def __getitem__(self, idx: Union[int, slice, Iterable]):
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

        if not isinstance(idx, (int, slice, Iterable)) or isinstance(idx, (str, bytes)):
            logger.error(
                "Invalid index type. Expected int, Iterable, or slice. "
                "i.e. `42`, `[0,1,2]`, `(11,32,101)` or `[start:end:step]`."
            )
            return None

        # If the index is a single integer, we'll cast to a list. This will ensure
        # that when we index into the dataframe, we always get a pandas.Series
        # back. This simplifies the retrieval logic and makes it easier to get
        # consistent return values.
        if isinstance(idx, int):
            idx = [idx]

        original_model_output = self.get_model_output(idx)
        return from_numpy(original_model_output)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data)

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
        """Get the metadata fields associated with the original dataset used to generate this one

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

    def _resolve_results_dir(self, results_dir: Optional[Union[Path, str]], verb: Optional[str]) -> Path:
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
                logger.info(msg)

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

        log_runtime_config(self.original_dataset_config, self.result_dir, ORIGINAL_DATASET_CONFIG_FILENAME)

    def write_batch(self, ids: np.ndarray, model_output: list[np.ndarray]):
        """Write a batch of model_output numpy arrays into a parquet results file.
        This writes the whole batch immediately.

        Caller is in charge of batch size consistency. It is expected that the
        output will have a consistent shape and that `ids` is the same length as
        model_output.

        Parameters
        ----------
        ids : np.ndarray
            Array of IDs, dtype of the elements must match the dtype type of the
            ids of the original dataset used to construct this InferenceDataSetWriter.
        model_output : list[np.ndarray]
            List of consistently dimensioned numpy arrays to save.
        """
        _shape = json.dumps(model_output[0].shape)

        flattened_model_output = [o.flatten() for o in model_output]
        table = pa.Table.from_arrays([ids, flattened_model_output], names=["id", "model_output"])

        table = table.replace_schema_metadata({"model_output_shape": _shape})
        pq.write_table(table, self.result_dir / "output.parquet")
