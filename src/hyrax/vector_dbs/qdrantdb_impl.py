import os
import uuid
from typing import Union

import numpy as np
from qdrant_client import QdrantClient, models

from hyrax.vector_dbs.vector_db_interface import VectorDB


class QdrantDB(VectorDB):
    """Implementation of the VectorDB interface using Qdrant as the backend."""

    def __init__(self, config, context):
        super().__init__(config, context)
        self.client = None
        self.collection_size = 0

    def _convert_id_to_uuid(self, id: Union[str, int]) -> str:
        """Convert an id to a UUID string using the OID namespace."""
        return uuid.uuid5(uuid.NAMESPACE_OID, str(id)).urn

    def connect(self):
        """Connect to the Qdrant database and return an instance of the client."""
        # Results_dir is the directory where the Qdrant database is stored.
        results_dir = self.context["results_dir"]
        self.client = QdrantClient(path=results_dir)

        collections = self.client.get_collections().collections
        if len(collections):
            self.collection_name = collections[0].name
        else:
            self.collection_name = None

        return self.client

    def create(self):
        """Create a new Qdrant database"""
        if self.client is None:
            self.connect()

        # We'll get the number of collection that are in the db, but for now
        # we follow the advice of the documentation, and restrict the database
        # to a single collection.
        # https://qdrant.tech/documentation/concepts/collections/#setting-up-multitenancy
        self.collection_index = len(self.client.get_collections().collections)

        # Note: Qdrant has an internal definition of "shard" that is different than
        # what is currently used by Hyrax (specifically ChromaDB). Here we set
        # shard_number to CPU core count * 2. Qdrant docs advocate starting with 12:
        # https://qdrant.tech/documentation/guides/distributed_deployment/#choosing-the-right-number-of-shards
        self.collection_name = f"shard_{self.collection_index}"
        created_collection = None
        if not self.client.collection_exists(self.collection_name):
            created_collection = self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    #! This stinks - we should just check the size of the data
                    #! when we call `save_to_database` and then set this automatically
                    #! as a parameter in self.context["blah"] or something.
                    size=self.config["vector_db"]["qdrant"]["vector_size"],
                    distance=models.Distance.EUCLID,
                    on_disk=True,
                ),
                shard_number=os.cpu_count() * 2,
            )

        if not created_collection:
            raise RuntimeError(f"Failed to create collection {self.collection_name} in Qdrant.")

        self.collection_size = self.client.count(collection_name=self.collection_name, exact=True)

        return self.collection_name

    def insert(self, ids: list[Union[str, int]], vectors: list[np.ndarray]):
        """Insert a batch of vectors into the Qdrant database.

        Parameters
        ----------
        ids : list[Union[str, int]]
            The ids to associate with the vectors
        vectors : list[np.ndarray]
            The vectors to insert into the database
        """
        if self.client is None:
            self.connect()

        expected_size = self.config["vector_db"]["qdrant"]["vector_size"]
        for idx, vector in enumerate(vectors):
            if len(vector) != expected_size:
                raise ValueError(
                    f"Vector at index {idx} has size {len(vector)}, but expected size is {expected_size}."
                )

        uuids = [self._convert_id_to_uuid(i) for i in ids]
        # Insert data into the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=uuids,
                vectors=vectors,
                payloads=[{"id": id} for id in ids],
            ),
        )

        # Update the collection size after insertion
        self.collection_size = self.client.count(collection_name=self.collection_name, exact=True).count
        return self.collection_size

    def search_by_id(self, id: Union[str, int], k: int = 1) -> dict[int, list[Union[str, int]]]:
        """Get the ids of the k nearest neighbors for a given id in the database.

        Qdrant will exclude the id itself from the results, thus we first
        retrieve the vector for a given id, and then use that vector to find the
        k nearest neighbors.

        Parameters
        ----------
        id : Union[str, int]
            The id of the vector in the database for which we want to find the
            k nearest neighbors
        k : int, optional
            The number of nearest neighbors to return, by default 1, return only
            the closest neighbor

        Returns
        -------
        dict[int, list[Union[str, int]]]
            Dictionary with input vector id as the key and the ids of the k
            nearest neighbors as the value.
        """
        if self.client is None:
            self.connect()

        # Retrieve the vector for the given id
        query_vector = self.get_by_id(id)

        # Find the k nearest neighbors for that vector
        res = {id: self._query_by_vector(query_vector[id], k)}
        return res

    def search_by_vector(
        self, vectors: Union[np.ndarray, list[np.ndarray]], k: int = 1
    ) -> dict[int, list[Union[str, int]]]:
        """Get the ids of the k nearest neighbors for a given vector.

        Parameters
        ----------
        vectors : Union[np.array, list[np.ndarray]]
            The one or more vectors to use when searching for nearest neighbors
        k : int, optional
            The number of nearest neighbors to return, by default 1, return only
            the closest neighbor

        Returns
        -------
        dict[int, list[Union[str, int]]]
            Dictionary with input vector index as the key and the ids of the
            k nearest neighbors as the value.
        """
        if self.client is None:
            self.connect()

        # If a single vector is provided, convert it to a list
        if isinstance(vectors, np.ndarray):
            vectors = [vectors]

        # Find the k nearest neighbors for the provided vector
        res = {i: self._query_by_vector(v, k) for i, v in enumerate(vectors)}
        return res

    def _query_by_vector(self, vector: np.ndarray, k: int = 1) -> list[str]:
        """Query the Qdrant database for the k nearest neighbors of a given vector."""

        query_results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.NearestQuery(nearest=vector),
            search_params=models.SearchParams(),
            limit=k,
        )

        return [point.payload["id"] for point in query_results.points]

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
        if self.client is None:
            self.connect()

        if not isinstance(ids, list):
            ids = [ids]

        uuids = [self._convert_id_to_uuid(i) for i in ids]

        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=uuids,
            with_vectors=True,
            with_payload=True,
        )

        # Return the vectors for the given ids
        return {point.payload["id"]: point.vector for point in points}
