import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import numpy as np
from pymilvus import MilvusClient

from hyrax.vector_dbs.vector_db_interface import VectorDB

MIN_SHARDS_FOR_PARALLELIZATION = 50


def _query_for_nn(results_dir: str, shard_name: str, vectors: list[np.ndarray], k: int):
    """The query function for the ProcessPoolExecutor to query a shard for the
    nearest neighbors of a set of vectors.

    Parameters
    ----------
    results_dir : str
        The directory where the ChromaDB results are stored
    shard_name : str
        The name of the ChromaDB shard to load and query
    vectors : np.ndarray
        The vectors used as inputs for the nearest neighbor search
    k : int
        The number of nearest neighbors to return

    Returns
    -------
    dict
        The results of the nearest neighbor search for the given vectors in the
        given shard.
    """
    database_file_name = Path(results_dir) / "milvus.db"
    milvusdb_client = MilvusClient(str(database_file_name))
    collection = milvusdb_client.get_collection(name=shard_name)
    return collection.query(query_embeddings=vectors, n_results=k)


def _query_for_id(results_dir: str, shard_name: str, id: str):
    """The query function for the ProcessPoolExecutor to query a shard for the
    vector associated with a given id.

    Parameters
    ----------
    results_dir : str
        The directory where the ChromaDB results are stored
    shard_name : str
        The name of the ChromaDB shard to load and query
    id : str
        The id of the vector in the database shard we are trying to retrieve

    Returns
    -------
    dict
        The results of the id query for the given id in the given shard.
    """
    database_file_name = Path(results_dir) / "milvus.db"
    milvusdb_client = MilvusClient(str(database_file_name))
    collection = milvusdb_client.get_collection(name=shard_name)
    return collection.get(id=id, output_fields=["vector"])


class MilvusDB(VectorDB):
    """Implementation of the VectorDB interface using MilvusDB as the backend."""

    def __init__(self, config, context):
        super().__init__(config, context)
        self.milvusdb_client = None
        self.collection = None

        self.shard_index = 0  # The current shard id for insertion
        self.shard_size = 0  # The number of vectors in the current shard

        # The approximate maximum size of a shard before a new one is created
        self.shard_size_limit = 65_536

        self.min_shards_for_parallelization = MIN_SHARDS_FOR_PARALLELIZATION

    def connect(self):
        """Create a database connection"""
        results_dir = self.context["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        database_file_name = Path(results_dir) / "milvus.db"
        self.milvusdb_client = MilvusClient(str(database_file_name))
        return self.milvusdb_client

    def create(self):
        """Create a new database"""

        if self.milvusdb_client is None:
            self.connect()

        self.shard_index = len(self.milvusdb_client.list_collections())

        # Create a chromadb shard (a.k.a. "collection")
        self.collection = f"shard_{self.shard_index}"
        self.milvusdb_client.create_collection(
            collection_name=self.collection,
            dimension=3,
            metric_type="L2",
        )

        return self.collection

    def insert(self, ids: list[Union[str | int]], vectors: list[np.ndarray]):
        """Insert a batch of vectors into the database.

        Parameters
        ----------
        ids : list[Union[str | int]]
            The ids to associate with the vectors
        vectors : list[np.ndarray]
            The vectors to insert into the database
        """

        # increment counter, if exceeds shard limit, create a new collection
        self.shard_size += len(ids)
        if self.shard_size > self.shard_size_limit:
            self.shard_size = len(ids)
            print("Creating new shard")
            self.create()

        try:
            int_ids = [int(id) for id in ids]
        except ValueError:
            raise ValueError("All ids must be convertible to int")

        self.milvusdb_client.insert(
            collection_name=self.collection,
            data=[{"id": id, "vector": vector} for id, vector in zip(int_ids, vectors)],
        )

    def search_by_id(self, id: Union[str | int], k: int = 1) -> dict[int, list[Union[str | int]]]:
        """Get the ids of the k nearest neighbors for a given id in the database.

        Parameters
        ----------
        id : Union[str | int]
            The id of the vector in the database for which we want to find the
            k nearest neighbors. If type `int` is provided, it will be converted
            to a string.
        k : int, optional
            The number of nearest neighbors to return. By default 1, return only
            the closest neighbor - this is almost always the same as the input.

        Returns
        -------
        dict[int, list[Union[str, int]]]
            Dictionary with input id index as the key and the ids of the k
            nearest neighbors as the value. Because this function accepts only 1
            id, the key will always be 0. i.e. {0: [id1, id2, ...]}

        Raises
        ------
        ValueError
            If more than one vector is found for the given id
        """

        if k < 1:
            raise ValueError("k must be greater than 0")

        # create the database connection
        if self.milvusdb_client is None:
            self.connect()

        if isinstance(id, int):
            id = str(id)

        # get all the shards
        shards = self.milvusdb_client.list_collections()

        vectors = []

        # ~ ProcessPoolExecutor parallelized
        if len(shards) > MIN_SHARDS_FOR_PARALLELIZATION:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(_query_for_id, self.context["results_dir"], shard.name, id): shard
                    for shard in shards
                }
                for future in as_completed(futures):
                    results = future.result()
                    vectors.extend(results["embeddings"])

        # ~ Non-parallelized implementation, faster for smaller number of shards
        else:
            # Query each shard, return vector for the given id.
            for shard in shards:
                # Get the vector for the id
                collection = self.milvusdb_client.get_collection(name=shard.name)
                results = collection.get(id, include=["embeddings"])
                vectors.extend(results["embeddings"])

        query_results: dict[int, list[Union[str | int]]] = {}
        # no matching id found in database
        if len(vectors) == 0:
            query_results = {}

        # multiple matching ids found in database
        elif len(vectors) > 1:
            raise ValueError(f"More than one vector found for id: {id}")

        # single matching id found in database
        else:
            query_results = self.search_by_vector(vectors, k=k)

        return query_results

    def search_by_vector(self, vectors: list[np.ndarray], k: int = 1) -> dict[int, list[Union[str | int]]]:
        """Get the ids of the k nearest neighbors for a given vector.

        Parameters
        ----------
        vectors : np.ndarray
            The vector to use when searching for nearest neighbors
        k : int, optional
            The number of nearest neighbors to return, by default 1, return only
            the closest neighbor

        Returns
        -------
        dict[int, list[Union[str, int]]]
            Dictionary with input vector index as the key and the ids of the k
            nearest neighbors as the value.
        """

        if k < 1:
            raise ValueError("k must be greater than 0")

        # create the database connection
        if self.milvusdb_client is None:
            self.connect()

        # get all the shards
        shards = self.milvusdb_client.list_collections()

        # This dictionary will hold the k nearest neighbors ids for each input vector
        result_dict: dict[int, list[Union[str | int]]] = {i: [] for i in range(len(vectors))}

        # Intermediate results holds all of the query results from all shards.
        # These results will be sorted and trimmed to the appropriate length before
        # being added to `result_dict`.
        intermediate_results: dict[int, dict[str, list[Union[str | int]]]] = {
            i: {"ids": [], "distances": []} for i in range(len(vectors))
        }

        # ~ ProcessPoolExecutor parallelized
        if len(shards) > MIN_SHARDS_FOR_PARALLELIZATION:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(_query_for_nn, self.context["results_dir"], shard.name, vectors, k): shard
                    for shard in shards
                }
                for future in as_completed(futures):
                    results = future.result()
                    for i in range(len(results["ids"])):
                        intermediate_results[i]["ids"].extend(results["ids"][i])
                        intermediate_results[i]["distances"].extend(results["distances"][i])

        # ~ Non-parallelized implementation, faster for smaller number of shards
        else:
            # Query each shard, return the k nearest neighbors from each shard.
            for shard in shards:
                collection = self.milvusdb_client.get_collection(name=shard.name)
                results = collection.query(query_embeddings=vectors, n_results=k)
                for i in range(len(results["ids"])):
                    intermediate_results[i]["ids"].extend(results["ids"][i])
                    intermediate_results[i]["distances"].extend(results["distances"][i])

        # Sort the distances ascending
        for i in range(len(intermediate_results)):
            sorted_indicies = np.argsort(intermediate_results[i]["distances"], stable=True)
            result_dict[i] = [intermediate_results[i]["ids"][j] for j in sorted_indicies][:k]

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
        # create the database connection
        if self.milvusdb_client is None:
            self.connect()

        shards = self.milvusdb_client.list_collections()
        vectors = {}

        if len(shards) > self.min_shards_for_parallelization:
            import multiprocessing

            multiprocessing.set_start_method("spawn", force=True)
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(_query_for_id, self.context["results_dir"], shard, ids): shard
                    for shard in shards
                }
                for future in as_completed(futures):
                    results = future.result()
                    for indx, result_id in enumerate(results["ids"]):
                        vectors[result_id] = results["embeddings"][indx]

        else:
            for shard in shards:
                collection = self.milvusdb_client.get_collection(shard.name)
                results = collection.get(ids, include=["embeddings"])

                for indx, result_id in enumerate(results["ids"]):
                    vectors[result_id] = results["embeddings"][indx]

        return vectors
