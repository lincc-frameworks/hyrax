"""Lance-based vector database implementation.

LanceDB opens the existing Lance inference results table (written by the ``infer``
verb) and creates an HNSW index on it in-place.  Set ``vector_db_dir`` to the same
path as ``infer_results_dir`` so that both verbs operate on the same Lance dataset.
"""

import logging
import os
from pathlib import Path
from typing import Union

import numpy as np

from hyrax.vector_dbs.vector_db_interface import VectorDB

# Suppress Lance's Rust-level WARN messages (normal during index creation)
if "LANCE_LOG" not in os.environ:
    os.environ["LANCE_LOG"] = "error"

logger = logging.getLogger(__name__)

# These must match the constants in result_dataset.py
_LANCE_DB_DIR = "lance_db"
_TABLE_NAME = "results"


class LanceDB(VectorDB):
    """Implementation of the VectorDB interface using Lance as the backend.

    Unlike ChromaDB and QdrantDB which create a new store and fill it via
    :meth:`insert`, ``LanceDB`` opens the existing Lance inference results table
    (written by the ``infer`` verb) and creates an HNSW index on it in-place.

    Set ``vector_db_dir`` to the same directory as ``infer_results_dir`` so that
    :class:`~hyrax.verbs.save_to_database.SaveToDatabase` operates on the Lance
    dataset that already contains inference results.
    """

    def __init__(self, config, context):
        super().__init__(config, context)
        self.db = None
        self.table = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def connect(self):
        """Open the existing Lance inference results table.

        Returns
        -------
        lancedb.table.LanceTable
            The opened Lance table.
        """
        import lancedb

        results_dir = Path(self.context["results_dir"])
        lance_dir = results_dir / _LANCE_DB_DIR
        self.db = lancedb.connect(str(lance_dir))
        self.table = self.db.open_table(_TABLE_NAME)
        return self.table

    def create(self):
        """Create an HNSW index on the Lance results table.

        The table must already exist and be populated (written by the ``infer``
        verb).  If an index on the ``data`` column already exists it is replaced
        (``replace=True`` is the Lance default), making this call idempotent.

        Returns
        -------
        lancedb.table.LanceTable
            The indexed Lance table.
        """
        if self.table is None:
            self.connect()

        lance_cfg = self.config["vector_db"]["lance"]

        metric = lance_cfg["metric"]
        # TOML uses `false` as a sentinel for "not set"
        num_partitions = lance_cfg["num_partitions"]
        if num_partitions is False:
            num_partitions = None

        num_sub_vectors = lance_cfg["num_sub_vectors"]
        if num_sub_vectors is False:
            num_sub_vectors = None

        self.table.create_index(
            metric=metric.lower(),
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="data",
        )
        logger.info("Lance HNSW index created on 'data' column (metric=%s)", metric)
        return self.table

    def insert(self, ids: list[Union[str, int]], vectors: list[np.ndarray]):
        """Insert new vectors into the Lance table, skipping duplicate IDs.

        Parameters
        ----------
        ids : list[Union[str, int]]
            The ids to associate with the vectors.
        vectors : list[np.ndarray]
            The vectors to insert into the database.
        """
        import pyarrow as pa

        if self.table is None:
            self.connect()

        # Deduplicate: skip IDs that are already present
        existing = self._get_existing_ids(ids)
        mask = [i for i in range(len(ids)) if ids[i] not in existing]
        new_ids = [ids[i] for i in mask]
        new_vectors = [vectors[i] for i in mask]

        if not new_ids:
            return

        data_type = self.table.schema.field("data").type
        arrow_table = pa.table(
            {
                "object_id": pa.array([str(i) for i in new_ids], type=pa.string()),
                "data": pa.array([v.flatten().tolist() for v in new_vectors], type=data_type),
            }
        )
        self.table.add(arrow_table)

    def search_by_vector(
        self, vectors: Union[np.ndarray, list[np.ndarray]], k: int = 1
    ) -> dict[int, list[Union[str, int]]]:
        """Get the IDs of the k nearest neighbors for one or more query vectors.

        Parameters
        ----------
        vectors : Union[np.ndarray, list[np.ndarray]]
            One or more query vectors.
        k : int, optional
            Number of nearest neighbors to return, by default 1.

        Returns
        -------
        dict[int, list[Union[str, int]]]
            Dictionary with input vector index as key and neighbor IDs as value.
        """
        if self.table is None:
            self.connect()

        if isinstance(vectors, np.ndarray) and vectors.ndim == 1:
            vectors = [vectors]
        elif isinstance(vectors, np.ndarray):
            vectors = list(vectors)

        results = {}
        for i, vector in enumerate(vectors):
            hits = self.table.search(vector, vector_column_name="data").limit(k).to_list()
            results[i] = [hit["object_id"] for hit in hits]
        return results

    def search_by_id(self, id: Union[str, int], k: int = 1) -> dict[int, list[Union[str, int]]]:
        """Get the IDs of the k nearest neighbors for the vector stored under ``id``.

        Parameters
        ----------
        id : Union[str, int]
            The ID of the query vector in the database.
        k : int, optional
            Number of nearest neighbors to return, by default 1.

        Returns
        -------
        dict[Union[str, int], list[Union[str, int]]]
            Dictionary with the input ID as key and neighbor IDs as value.
        """
        vector_map = self.get_by_id([id])
        if id not in vector_map:
            raise KeyError(f"ID {id!r} not found in Lance table")
        results = self.search_by_vector([np.array(vector_map[id])], k=k)
        return {id: results[0]}

    def get_by_id(self, ids: list[Union[str, int]]) -> dict[Union[str, int], list[float]]:
        """Retrieve vectors by their IDs.

        Parameters
        ----------
        ids : list[Union[str, int]]
            IDs of vectors to retrieve.

        Returns
        -------
        dict[Union[str, int], list[float]]
            Dictionary mapping each found ID to its vector.
        """
        if self.table is None:
            self.connect()

        # Accept a single id as well as a list.
        if not isinstance(ids, list):
            ids = [ids]
        str_ids = [str(i) for i in ids]
        id_list = ", ".join(f"'{_escape_sql(sid)}'" for sid in str_ids)
        rows = self.table.search().where(f"object_id IN ({id_list})", prefilter=True).to_list()
        return {row["object_id"]: list(row["data"]) for row in rows}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_existing_ids(self, ids: list[Union[str, int]]) -> set[Union[str, int]]:
        """Return the subset of *ids* that already exist in the table."""
        str_ids = [str(i) for i in ids]
        id_list = ", ".join(f"'{_escape_sql(sid)}'" for sid in str_ids)
        rows = self.table.search().where(f"object_id IN ({id_list})", prefilter=True).to_list()
        return {row["object_id"] for row in rows}


def _escape_sql(value: str) -> str:
    """Escape a string value for use in a SQL single-quoted literal."""
    return value.replace("'", "''")
