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

    def connect(self):
        """Connect to the Qdrant database and return an instance of the client."""
        # Results_dir is the directory where the Qdrant database is stored.
        results_dir = self.context["results_dir"]
        self.client = QdrantClient(path=results_dir)
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
        # shard_number to 12 based on documentation for when you "anticipate a
        # lot of growth".
        # https://qdrant.tech/documentation/guides/distributed_deployment/#choosing-the-right-number-of-shards
        self.collection_name = f"shard_{0}"
        if not self.client.collection_exists(self.collection_name):
            created_collection = self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    #! This stinks - we should just check the size of the data
                    #! when we call `save_to_database` and then set this automatically
                    #! as a parameter in self.context["blah"] or something.
                    size=self.config["vector_db.qdrant"]["vector_size"],
                    distance=models.Distance.DOT,
                    on_disk=True,
                ),
                shard_number=12,
            )

        if not created_collection:
            raise RuntimeError(f"Failed to create collection {self.collection_name} in Qdrant.")

        self.collection_size = self.client.count(collection_name=self.collection_name, exact=True)

        return self.collection_name

    def insert(self, ids: list[Union[str, int]], vectors: list[np.ndarray]):
        """Insert data into the Qdrant database."""
        if self.client is None:
            self.connect()

        #! Insert a check here to make sure that the vectors are the same
        #! length as what is specified for the collection!!!

        #! Insert a check here to make sure that the ids are UUIDs _or_ integers!!!
        #! We need to find a way to make string input convert to UUID
        """
        if not isinstance(ids[0], int):
            import uuid
            uuids = [str(uuid.uuid5(i)) for i in ids]
        """

        # Insert data into the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
            ),
        )

        # Update the collection size after insertion
        self.collection_size = self.client.count(collection_name=self.collection_name, exact=True).count
        return self.collection_size

    def search_by_id(self, id: Union[str, int], k: int = 1) -> dict[int, list[Union[str, int]]]:
        """Search for the k nearest neighbors of a given id. It appears that Qdrant
        will exclude the id itself from the results. Thus we first retrieve the
        vector for the given id, and then use that vector to find the k nearest
        neighbors."""
        if self.client is None:
            self.connect()

        # Retrieve the vector for the given id
        query_vector = self.get_by_id(id)

        # Find the k nearest neighbors using the vector
        query_results = self.client.query_points(
            collection_name=self.collection_name, query=models.NearestQuery(nearest=query_vector[id]), limit=k
        )

        # Return the ids of the k nearest neighbors
        return {id: [p.id for p in query_results.points]}

    def search_by_vector(
        self, vectors: Union[np.ndarray, list[np.ndarray]], k: int = 1
    ) -> dict[int, list[Union[str, int]]]:
        """Search for the k nearest neighbors of a given vector."""
        if self.client is None:
            self.connect()

        # If a single vector is provided, convert it to a list
        if isinstance(vectors, np.ndarray):
            vectors = [vectors]

        # Search for the k nearest neighbors
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vectors[0],
            limit=k,
        )

        # Return the ids of the k nearest neighbors
        return {0: [hit.id for hit in search_result]}

    def get_by_id(self, ids: list[Union[str, int]]) -> dict[Union[str, int], list[float]]:
        """Get the vectors for a list of ids."""
        if self.client is None:
            self.connect()

        if not isinstance(ids, list):
            ids = [ids]

        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_vectors=True,
        )

        # Return the vectors for the given ids
        return {point.id: point.vector for point in points}
