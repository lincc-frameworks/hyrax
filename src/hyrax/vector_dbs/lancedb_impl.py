import logging
from typing import Union

import lancedb
import numpy as np

from hyrax.vector_dbs.vector_db_interface import VectorDB

logger = logging.getLogger()


class LanceDB(VectorDB):
    """Implementation of the VectorDB interface using LanceDB as the backend."""

    def __init__(self, config, context):
        super().__init__(config, context)
        self.db = None
        self.table = None
        self.table_name = "vectors"

    def connect(self):
        """Connect to the LanceDB database and return an instance of the client.

        Uses lancedb.connect() to establish a connection to the database directory.
        """
        results_dir = self.context["results_dir"]
        self.db = lancedb.connect(results_dir)

        # Check if table already exists
        table_names = self.db.table_names()
        if self.table_name in table_names:
            self.table = self.db.open_table(self.table_name)

        return self.db

    def create(self):
        """Create a new LanceDB table for storing vectors.

        Creates a table with schema for id and vector embeddings.
        """
        if self.db is None:
            self.connect()

        # Check if table already exists
        table_names = self.db.table_names()
        if self.table_name not in table_names:
            # Create an empty table with the first insert
            # LanceDB requires at least one row to create a table
            self.table = None
        else:
            self.table = self.db.open_table(self.table_name)

        return self.table

    def insert(self, ids: list[Union[str, int]], vectors: list[np.ndarray]):
        """Insert a batch of vectors into the LanceDB table.

        Parameters
        ----------
        ids : list[Union[str, int]]
            The ids to associate with the vectors
        vectors : list[np.ndarray]
            The vectors to insert into the database
        """
        if self.db is None:
            self.connect()

        # Convert ids to strings for consistency
        str_ids = [str(id) for id in ids]

        # Prepare data in the format LanceDB expects
        data = [{"id": str_id, "vector": vector.tolist()} for str_id, vector in zip(str_ids, vectors)]

        # Create or append to table
        if self.table is None:
            # First insert - create the table
            self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")
        else:
            # Subsequent inserts - append to existing table
            # Check for duplicates and filter them out
            try:
                existing_data = self.table.to_pandas()
                existing_ids = set(existing_data["id"].tolist())
                data = [d for d in data if d["id"] not in existing_ids]

                if len(data) > 0:
                    self.table.add(data)
            except Exception:
                # If table is empty or doesn't exist yet, just add the data
                self.table.add(data)

    def search_by_id(self, id: Union[str, int], k: int = 1) -> dict[int, list[Union[str, int]]]:
        """Get the ids of the k nearest neighbors for a given id in the database.

        Parameters
        ----------
        id : Union[str, int]
            The id of the vector in the database for which we want to find the
            k nearest neighbors
        k : int, optional
            The number of nearest neighbors to return, by default 1

        Returns
        -------
        dict[int, list[Union[str, int]]]
            Dictionary with input id as the key and the ids of the k
            nearest neighbors as the value.
        """
        if k < 1:
            raise ValueError("k must be greater than 0")

        if self.db is None:
            self.connect()

        if self.table is None:
            return {}

        # Convert id to string for consistency
        str_id = str(id)

        # Get the vector for the given id
        vectors = self.get_by_id([str_id])

        if not vectors or str_id not in vectors:
            return {}

        # Use the vector to search for nearest neighbors
        vector = vectors[str_id]
        results = self.search_by_vector([np.array(vector)], k=k)

        # Return in the expected format
        return {id: results[0]}

    def search_by_vector(
        self, vectors: Union[np.ndarray, list[np.ndarray]], k: int = 1
    ) -> dict[int, list[Union[str, int]]]:
        """Get the ids of the k nearest neighbors for a given vector.

        Parameters
        ----------
        vectors : Union[np.ndarray, list[np.ndarray]]
            The vector(s) to use when searching for nearest neighbors
        k : int, optional
            The number of nearest neighbors to return, by default 1

        Returns
        -------
        dict[int, list[Union[str, int]]]
            Dictionary with input vector index as the key and the ids of the k
            nearest neighbors as the value.
        """
        if k < 1:
            raise ValueError("k must be greater than 0")

        if self.db is None:
            self.connect()

        if self.table is None:
            return {}

        # Ensure vectors is a list
        if isinstance(vectors, np.ndarray):
            vectors = [vectors]

        result_dict = {}

        for i, vector in enumerate(vectors):
            # LanceDB search returns results ordered by distance
            search_results = self.table.search(vector.tolist()).limit(k).to_pandas()

            # Extract the ids from the results
            neighbor_ids = search_results["id"].tolist()
            result_dict[i] = neighbor_ids

        return result_dict

    def get_by_id(self, ids: list[Union[str, int]]) -> dict[Union[str, int], list[float]]:
        """Retrieve the vectors associated with a list of ids.

        Parameters
        ----------
        ids : list[Union[str, int]]
            The ids of the vectors to retrieve.

        Returns
        -------
        dict[Union[str, int], list[float]]
            Dictionary with the ids as the keys and the vectors as the values.
        """
        if self.db is None:
            self.connect()

        if self.table is None:
            return {}

        # Convert all ids to strings for consistency
        str_ids = [str(id) for id in ids]

        # Query the table for the given ids
        df = self.table.to_pandas()

        result = {}
        for str_id, original_id in zip(str_ids, ids):
            matching_rows = df[df["id"] == str_id]
            if not matching_rows.empty:
                vector = matching_rows.iloc[0]["vector"]
                # Return with original id type
                result[original_id] = vector

        return result
