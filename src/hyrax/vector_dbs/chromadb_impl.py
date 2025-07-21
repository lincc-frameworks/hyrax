import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Union

import chromadb
import numpy as np

from hyrax.vector_dbs.vector_db_interface import VectorDB

MIN_SHARDS_FOR_PARALLELIZATION = 50

logger = logging.getLogger()


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
    chromadb_client = chromadb.PersistentClient(path=str(results_dir))
    collection = chromadb_client.get_collection(name=shard_name)
    return collection.query(query_embeddings=vectors, n_results=k)


def _query_for_id(results_dir: str, shard_name: str, id: Union[str, list[str]], include: Optional[list[str]]):
    """The query function for the ProcessPoolExecutor to query a shard for the
    vector associated with a given id.

    Parameters
    ----------
    results_dir : str
        The directory where the ChromaDB results are stored
    shard_name : str
        The name of the ChromaDB shard to load and query
    id : Union[str, list[str]]
        One or more ids of vectors in the database shard we are trying to retrieve
    include : list[str], optional
        The fields to include in the results.

    Returns
    -------
    dict
        The results of the query for the given ids in the given shard.
    """
    chromadb_client = chromadb.PersistentClient(path=str(results_dir))
    collection = chromadb_client.get_collection(name=shard_name)
    if not include:
        include = ["embeddings"]
    return collection.get(id, include=include)


class ChromaDB(VectorDB):
    """Implementation of the VectorDB interface using ChromaDB as the backend."""

    def __init__(self, config, context):
        super().__init__(config, context)
        self.chromadb_client = None
        self.collection = None

        self.shard_index = 0  # The current shard id for insertion
        self.shard_size = 0  # The number of vectors in the current shard

        # The approximate maximum size of a shard before a new one is created
        self.shard_size_limit = self.config["vector_db"]["chromadb"]["shard_size_limit"]

        # If set, inserting a vector with number of elements >= this logs a warning.
        self.vector_size_limit = self.config["vector_db"]["chromadb"]["vector_size_warning"]

        # Min number of shards before using multiprocess to parallelize the search
        self.min_shards_for_parallelization = MIN_SHARDS_FOR_PARALLELIZATION

    def connect(self):
        """Create a database connection"""
        results_dir = self.context["results_dir"]
        self.chromadb_client = chromadb.PersistentClient(path=str(results_dir))
        return self.chromadb_client

    def create(self):
        """Create a new database"""

        if self.chromadb_client is None:
            self.connect()

        # If this database already has collections, we'll attempt to identify
        # the latests shard that was created.
        self.shard_index = len(self.chromadb_client.list_collections())

        # Create (or get) a chromadb shard (a.k.a. "collection")
        self.collection = self.chromadb_client.get_or_create_collection(
            name=f"shard_{self.shard_index}",
            metadata={
                # These are chromadb defaults, may want to make them configurable
                "hsnw:space": "l2",
                "hsnw:construction_ef": 100,
                "hsnw:search_ef": 100,
            },
        )

        # If the shard was pre-existing, we'll get the current number of records
        self.shard_size = self.collection.count()

        return self.collection

    def insert(self, ids: list[Union[str, int]], vectors: list[np.ndarray]):
        """Insert a batch of vectors into the database.

        Parameters
        ----------
        ids : list[Union[str | int]]
            The ids to associate with the vectors
        vectors : list[np.ndarray]
            The vectors to insert into the database
        """

        # Check to see if the ids we're about to insert are already in the database
        pre_existing_ids = self._get_ids(ids=ids)

        # create a mask, so that we don't insert vectors that are already present in the database
        mask = [i for i in range(len(ids)) if ids[i] not in pre_existing_ids]
        ids = [ids[i] for i in mask]
        vectors = [vectors[i] for i in mask]

        if len(ids) == 0:
            # no new vectors to insert
            return

        if self.vector_size_limit and len(vectors[0]) >= self.vector_size_limit:
            logger.warning(
                f"Attempting to insert vectors with length: {len(vectors[0])}.\
                           Chroma DB often has poor performance when working with vectors\
                           larger than {self.config['vector_db']['chromadb']['vector_size_warning']}"
            )

        # increment counter, if exceeds shard limit, create a new collection
        self.shard_size += len(ids)
        if self.shard_size > self.shard_size_limit:
            self.collection = self.create()
            self.shard_size = len(ids)

        self.collection.add(ids=ids, embeddings=vectors)

    def search_by_id(self, id: Union[str, int], k: int = 1) -> dict[int, list[Union[str, int]]]:
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
            Dictionary with input id as the key and the ids of the k
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
        if self.chromadb_client is None:
            self.connect()

        if isinstance(id, int):
            id = str(id)

        # get all the shards
        shards = self.chromadb_client.list_collections()

        vectors = []

        # ~ ProcessPoolExecutor parallelized
        if len(shards) > self.min_shards_for_parallelization:
            import multiprocessing

            multiprocessing.set_start_method("spawn", force=True)
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        _query_for_id, self.context["results_dir"], shard.name, id, ["embeddings"]
                    ): shard
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
                collection = self.chromadb_client.get_collection(name=shard.name)
                results = collection.get(id, include=["embeddings"])
                vectors.extend(results["embeddings"])

        query_results: dict[int, list[Union[str, int]]] = {}
        # no matching id found in database
        if len(vectors) == 0:
            query_results = {}

        # multiple matching ids found in database
        elif len(vectors) > 1:
            raise ValueError(f"More than one vector found for id: {id}")

        # single matching id found in database
        else:
            query_results = self.search_by_vector(vectors, k=k)

        # Return the dictionary as {id: neighbor_ids}
        return {id: query_results[0]}

    def search_by_vector(
        self, vectors: Union[np.ndarray, list[np.ndarray]], k: int = 1
    ) -> dict[int, list[Union[str, int]]]:
        """Get the ids of the k nearest neighbors for a given vector.

        Parameters
        ----------
        vectors : Union[np.ndarray, list[np.ndarray]]
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
        if self.chromadb_client is None:
            self.connect()

        # get all the shards
        shards = self.chromadb_client.list_collections()

        # This dictionary will hold the k nearest neighbors ids for each input vector
        result_dict: dict[int, list[Union[str, int]]] = {i: [] for i in range(len(vectors))}

        # Intermediate results holds all of the query results from all shards.
        # These results will be sorted and trimmed to the appropriate length before
        # being added to `result_dict`.
        intermediate_results: dict[int, dict[str, list[Union[str, int]]]] = {
            i: {"ids": [], "distances": []} for i in range(len(vectors))
        }

        # ~ ProcessPoolExecutor parallelized
        if len(shards) > self.min_shards_for_parallelization:
            import multiprocessing

            multiprocessing.set_start_method("spawn", force=True)
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
                collection = self.chromadb_client.get_collection(name=shard.name)
                results = collection.query(query_embeddings=vectors, n_results=k)
                for i in range(len(results["ids"])):
                    intermediate_results[i]["ids"].extend(results["ids"][i])
                    intermediate_results[i]["distances"].extend(results["distances"][i])

        # Sort the distances ascending
        for i in range(len(intermediate_results)):
            sorted_indicies = np.argsort(intermediate_results[i]["distances"])
            result_dict[i] = [intermediate_results[i]["ids"][j] for j in sorted_indicies][:k]

        return result_dict

    def get_by_id(self, ids: list[Union[str, int]]) -> dict[Union[str, int], list[float]]:
        """Retrieve the vectors associated with a list of ids.

        Parameters
        ----------
        ids : list[Union[str, int]]
            The ids of the vectors to retrieve. For ChromaDB instances, these should
            always be strings.

        Returns
        -------
        dict[str, list[float]]
            Dictionary with the ids as the keys and the vectors as the values.
        """

        # create the database connection
        if self.chromadb_client is None:
            self.connect()

        shards = self.chromadb_client.list_collections()
        vectors = {}

        if len(shards) > self.min_shards_for_parallelization:
            import multiprocessing

            multiprocessing.set_start_method("spawn", force=True)
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        _query_for_id, self.context["results_dir"], shard.name, ids, ["embeddings"]
                    ): shard
                    for shard in shards
                }
                for future in as_completed(futures):
                    results = future.result()
                    for indx, result_id in enumerate(results["ids"]):
                        vectors[result_id] = results["embeddings"][indx]

        else:
            for shard in shards:
                collection = self.chromadb_client.get_collection(shard.name)
                results = collection.get(ids, include=["embeddings"])

                for indx, result_id in enumerate(results["ids"]):
                    vectors[result_id] = results["embeddings"][indx]

        return vectors

    def _get_ids(self, ids: list[Union[str, int]]) -> set[str]:
        """For the given list of ids, return the ids that are already in the database.

        Parameters
        ----------
        ids : list[Union[str, int]]
            The ids of the vectors to retrieve. For ChromaDB instances, these should
            always be strings.

        Returns
        -------
        set(str)
            Set of ids that are already in the database.
        """

        # create the database connection
        if self.chromadb_client is None:
            self.connect()

        shards = self.chromadb_client.list_collections()
        found_ids = set()

        if len(shards) > self.min_shards_for_parallelization:
            import multiprocessing

            multiprocessing.set_start_method("spawn", force=True)
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        _query_for_id, self.context["results_dir"], shard.name, ids, include=[]
                    ): shard
                    for shard in shards
                }
                for future in as_completed(futures):
                    results = future.result()
                    found_ids.update(results["ids"])

        else:
            for shard in shards:
                collection = self.chromadb_client.get_collection(shard.name)
                results = collection.get(ids, include=[])
                found_ids.update(results["ids"])

        return found_ids
