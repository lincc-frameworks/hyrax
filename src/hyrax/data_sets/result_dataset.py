"""Lance-based storage for inference results.

This module provides ResultDataset and ResultDatasetWriter classes that store
inference results in Lance columnar format instead of batched .npy files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Union

# Suppress Lance's Rust-level WARN about creating new datasets (normal on first write)
if "LANCE_LOG" not in os.environ:
    os.environ["LANCE_LOG"] = "error"

import lancedb
import numpy as np
import pyarrow as pa

from .data_set_registry import HyraxDataset

logger = logging.getLogger(__name__)

TABLE_NAME = "results"
LANCE_DB_DIR = "lance_db"


class ResultDatasetWriter:
    """Writer for Lance-based inference results.

    Writes inference results incrementally to Lance format using table.add()
    for each batch, avoiding memory accumulation.
    """

    def __init__(self, result_dir: Union[str, Path]):
        """Initialize the writer.

        Parameters
        ----------
        result_dir : Union[str, Path]
            Directory where Lance database will be created
        """
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.lance_dir = self.result_dir / LANCE_DB_DIR
        self.db = None
        self.table = None
        self.schema = None
        self.tensor_dtype = None
        self.tensor_shape = None
        self.batch_count = 0

    def write_batch(self, object_ids: np.ndarray, data: list[np.ndarray]):
        """Write a batch of results incrementally.

        Parameters
        ----------
        object_ids : np.ndarray
            Array of object IDs (will be converted to strings)
        data : list[np.ndarray]
            List of numpy arrays (tensors) to write
        """
        if len(object_ids) != len(data):
            raise ValueError("Length of object_ids must match length of data")

        if len(data) == 0:
            return

        # Convert data to numpy array for uniform handling
        data_array = np.array(data)
        first_tensor = data_array[0]

        # On first write, create the schema and table
        if self.schema is None:
            self._create_schema(first_tensor)
            self.db = lancedb.connect(str(self.lance_dir))
            # Create empty table with schema
            empty_data = pa.table(
                {
                    "object_id": pa.array([], type=pa.string()),
                    "data": pa.array([], type=self.schema.field("data").type),
                },
                schema=self.schema,
            )
            self.table = self.db.create_table(TABLE_NAME, empty_data, mode="overwrite")
        else:
            # Validate that all tensors match the established schema
            for i, tensor in enumerate(data):
                if tensor.dtype != self.tensor_dtype:
                    raise ValueError(
                        f"Tensor at index {i} has dtype {tensor.dtype}, "
                        f"but schema expects {self.tensor_dtype}"
                    )
                if tensor.shape != tuple(self.tensor_shape):
                    raise ValueError(
                        f"Tensor at index {i} has shape {tensor.shape}, "
                        f"but schema expects {tuple(self.tensor_shape)}"
                    )

        # Flatten tensors for storage
        flattened_data = [tensor.flatten() for tensor in data]

        # Create PyArrow record batch
        batch_data = {
            "object_id": pa.array([str(oid) for oid in object_ids], type=pa.string()),
            "data": pa.array(flattened_data, type=self.schema.field("data").type),
        }

        # Convert to PyArrow table and add to Lance
        arrow_table = pa.table(batch_data, schema=self.schema)
        self.table.add(arrow_table)
        self.batch_count += 1

        logger.debug(f"Wrote batch {self.batch_count} with {len(object_ids)} records")

    def commit(self):
        """Finalize the write by optimizing the table."""
        if self.table is not None:
            logger.info(f"Optimizing Lance table after {self.batch_count} batches")
            self.table.optimize()
            logger.info("Lance table optimization complete")

    def _create_schema(self, sample_tensor: np.ndarray):
        """Create PyArrow schema with tensor metadata.

        Parameters
        ----------
        sample_tensor : np.ndarray
            Sample tensor to determine dtype and shape
        """
        # Get dtype and shape from sample
        self.tensor_dtype = sample_tensor.dtype
        self.tensor_shape = list(sample_tensor.shape)
        flattened_size = int(np.prod(self.tensor_shape))

        # Map numpy dtype to PyArrow type
        pa_type = pa.from_numpy_dtype(self.tensor_dtype)

        # Create schema with metadata
        metadata = {
            b"tensor_shape": json.dumps(self.tensor_shape).encode("utf-8"),
            b"tensor_dtype": str(self.tensor_dtype).encode("utf-8"),
        }

        self.schema = pa.schema(
            [
                pa.field("object_id", pa.string()),
                pa.field("data", pa.list_(pa_type, flattened_size)),
            ],
            metadata=metadata,
        )

        logger.debug(
            f"Created schema for tensors with shape {self.tensor_shape} and dtype {self.tensor_dtype}"
        )


class ResultDataset(HyraxDataset):
    """Reader for Lance-based inference results.

    Provides HyraxQL-compatible getters to results stored in Lance format.
    """

    def __init__(self, config: dict, data_location: Union[Path, str]):
        """Initialize the dataset.

        Parameters
        ----------
        config : dict
            Hyrax configuration dictionary
        data_location : Union[Path, str]
            Path to results directory containing lance_db/
        """
        super().__init__(config)

        self.data_location = Path(data_location)
        self.lance_dir = self.data_location / LANCE_DB_DIR

        # Open Lance database and table
        if not self.lance_dir.exists():
            raise RuntimeError(f"Lance database directory {self.lance_dir} does not exist")

        self.db = lancedb.connect(str(self.lance_dir))
        self.table = self.db.open_table(TABLE_NAME)

        # Get the underlying lance dataset for efficient access
        self.lance_dataset = self.table.to_lance()

        # Get schema metadata
        schema_metadata = self.table.schema.metadata
        if schema_metadata is None:
            raise RuntimeError("Lance table schema is missing metadata")

        # Decode tensor shape and dtype from metadata
        self.tensor_shape = json.loads(schema_metadata[b"tensor_shape"].decode("utf-8"))
        self.tensor_dtype = np.dtype(schema_metadata[b"tensor_dtype"].decode("utf-8"))

        logger.debug(f"Opened Lance table with shape {self.tensor_shape} and dtype {self.tensor_dtype}")

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return self.table.count_rows()

    def __getitem__(self, idx: Union[int, np.ndarray]):
        """Get data by index.

        Parameters
        ----------
        idx : Union[int, np.ndarray]
            Single index or array of indices

        Returns
        -------
        np.ndarray
            Data tensor(s)

        Raises
        ------
        IndexError
            If index is out of range
        """
        # Handle single index
        is_single = isinstance(idx, (int, np.integer))
        if is_single:
            idx = [int(idx)]
        else:
            idx = np.asarray(idx)
            idx = [int(idx)] if len(idx.shape) == 0 else idx.tolist()  # scalar array

        # Validate indices
        table_len = len(self)
        for i in idx:
            if i < 0 or i >= table_len:
                raise IndexError(f"Index {i} is out of range for dataset of length {table_len}")

        # Use take for O(1) random access
        result = self.lance_dataset.take(idx)

        # Extract data column and reshape
        data_column = result["data"].to_pylist()
        tensors = []
        for flat_data in data_column:
            tensor = np.array(flat_data, dtype=self.tensor_dtype)
            tensor = tensor.reshape(self.tensor_shape)
            tensors.append(tensor)

        # Return single tensor or array of tensors
        return tensors[0] if is_single else np.array(tensors)

    def get_data(self, idx: int):
        """Get data tensor at index (HyraxQL getter).

        Parameters
        ----------
        idx : int
            Index of the data item

        Returns
        -------
        np.ndarray
            Data tensor
        """
        return self.__getitem__(idx)

    def get_object_id(self, idx: int) -> str:
        """Get object ID at index (HyraxQL getter).

        Parameters
        ----------
        idx : int
            Index of the data item

        Returns
        -------
        str
            Object ID
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of range for dataset of length {len(self)}")

        result = self.lance_dataset.take([idx])
        # Extract first row's object_id since we're taking a single index
        return result["object_id"][0].as_py()

    def ids(self) -> list[str]:
        """Generate all object IDs.

        Returns
        -------
        list[str]
            Object IDs in order
        """
        # Use scanner with projection to only read object_id column
        scanner = self.lance_dataset.scanner(columns=["object_id"])
        return [oid.as_py() for batch in scanner.to_batches() for oid in batch["object_id"]]
