import tempfile

import numpy as np

from hyrax.vector_dbs.chromadb_impl import ChromaDB


def test_connect():
    """Test that we can create a connections to the database"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chromadb_instance = ChromaDB({}, {"results_dir": temp_dir})
        chromadb_instance.connect()

        assert chromadb_instance.chromadb_client is not None


def test_create():
    """Test creation of a single collection (shard) in the database"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chromadb_instance = ChromaDB({}, {"results_dir": temp_dir})
        chromadb_instance.connect()
        chromadb_instance.create()

        collections = chromadb_instance.chromadb_client.list_collections()

        assert collections is not None
        assert len(collections) == 1
        assert collections[0].name == "shard_0"


def test_insert():
    """Ensure that we can insert IDs and vectors into the database"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chromadb_instance = ChromaDB({}, {"results_dir": temp_dir})
        chromadb_instance.connect()
        chromadb_instance.create()

        ids = ["id1", "id2"]
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        chromadb_instance.insert(ids, vectors)
        collection = chromadb_instance.chromadb_client.get_collection("shard_0")
        assert collection.count() == 2


def test_insert_creates_new_shards():
    """Ensure that we can insert IDs and vectors into the database, and that new
    shards are created when the shard size limit is reached"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chromadb_instance = ChromaDB({}, {"results_dir": temp_dir})
        chromadb_instance.shard_size_limit = 5
        chromadb_instance.min_shards_for_parallelization = 3
        chromadb_instance.connect()
        chromadb_instance.create()

        def random_vector_generator(batch_size=1):
            """Create random vectors"""
            while True:
                batch = [np.random.rand(3) for _ in range(batch_size)]
                yield batch

        batch_size = 2
        num_batches = 10

        vector_generator = random_vector_generator(batch_size * num_batches)
        ids = [str(i) for i in range(batch_size * num_batches)]
        vectors = [t.flatten() for t in next(vector_generator)]

        for i in range(num_batches):
            chromadb_instance.insert(
                ids=ids[batch_size * i : batch_size * (i + 1)],
                vectors=vectors[batch_size * i : batch_size * (i + 1)],
            )

        collections = chromadb_instance.chromadb_client.list_collections()
        assert len(collections) == 5


def test_search_by_id():
    """Test search_by_id retrieves nearest neighbor ids"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chromadb_instance = ChromaDB({}, {"results_dir": temp_dir})
        chromadb_instance.connect()
        chromadb_instance.create()

        ids = ["id1", "id2"]
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        chromadb_instance.insert(ids, vectors)

        # Search by single vector should return the id1 and id2 in that order
        result = chromadb_instance.search_by_id("id1", k=2)
        assert len(result[0]) == 2
        assert np.all(result[0] == ["id1", "id2"])

        # Search should return all ids when k is larger than the number of ids
        result = chromadb_instance.search_by_id("id1", k=5)
        assert len(result[0]) == 2
        assert np.all(result[0] == ["id1", "id2"])

        # Search should return 1 id when k is 1
        result = chromadb_instance.search_by_id("id1", k=1)
        assert len(result[0]) == 1
        assert np.all(result[0] == ["id1"])

        # Search by another vector should return the id2 and id1 in that order
        result = chromadb_instance.search_by_id("id2", k=2)
        assert len(result[0]) == 2
        assert np.all(result[0] == ["id2", "id1"])


def test_search_by_id_many_shards():
    """Test search_by_id retrieves nearest neighbor ids when there are many shards"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chromadb_instance = ChromaDB({}, {"results_dir": temp_dir})
        chromadb_instance.shard_size_limit = 5
        chromadb_instance.min_shards_for_parallelization = 3
        chromadb_instance.connect()
        chromadb_instance.create()

        def random_vector_generator(batch_size=1):
            """Create random vectors"""
            while True:
                batch = [np.random.rand(3) for _ in range(batch_size)]
                yield batch

        batch_size = 2
        num_batches = 10

        vector_generator = random_vector_generator(batch_size * num_batches)
        ids = [str(i) for i in range(batch_size * num_batches)]
        vectors = [t.flatten() for t in next(vector_generator)]

        for i in range(num_batches):
            chromadb_instance.insert(
                ids=ids[batch_size * i : batch_size * (i + 1)],
                vectors=vectors[batch_size * i : batch_size * (i + 1)],
            )

        ids = ["id1", "id2"]
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        chromadb_instance.insert(ids, vectors)

        # Search should return 1 id when k is 1
        result = chromadb_instance.search_by_id("id1", k=1)
        assert len(result[0]) == 1
        assert np.all(result[0] == ["id1"])


def test_search_by_vector():
    """Test search_by_vector retrieves nearest neighbor ids"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chromadb_instance = ChromaDB({}, {"results_dir": temp_dir})
        chromadb_instance.connect()
        chromadb_instance.create()

        ids = ["id1", "id2"]
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        chromadb_instance.insert(ids, vectors)

        # Search by single vector should return the id1 and id2 in that order
        result = chromadb_instance.search_by_vector([np.array([1, 2, 3])], k=2)
        assert len(result[0]) == 2
        assert np.all(result[0] == ["id1", "id2"])

        # Search should return all ids when k is larger than the number of ids
        result = chromadb_instance.search_by_vector([np.array([1, 2, 3])], k=5)
        assert len(result[0]) == 2
        assert np.all(result[0] == ["id1", "id2"])

        # Search should return 1 id when k is 1
        result = chromadb_instance.search_by_vector([np.array([1, 2, 3])], k=1)
        assert len(result[0]) == 1
        assert np.all(result[0] == ["id1"])

        # Search by another vector should return the id2 and id1 in that order
        result = chromadb_instance.search_by_vector([np.array([4, 5, 6])], k=2)
        assert len(result[0]) == 2
        assert np.all(result[0] == ["id2", "id1"])

        # Search by multiple vectors should return the ids in the order of the vectors
        result = chromadb_instance.search_by_vector([np.array([4, 5, 6]), np.array([1, 2, 3])], k=2)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2
        assert np.all(result[0] == ["id2", "id1"])
        assert np.all(result[1] == ["id1", "id2"])


def test_search_by_vector_many_shards():
    """Test search_by_vector retrieves nearest neighbor ids when there are many shards"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chromadb_instance = ChromaDB({}, {"results_dir": temp_dir})
        chromadb_instance.shard_size_limit = 5
        chromadb_instance.min_shards_for_parallelization = 3
        chromadb_instance.connect()
        chromadb_instance.create()

        def random_vector_generator(batch_size=1):
            """Create random vectors"""
            while True:
                batch = [np.random.rand(3) for _ in range(batch_size)]
                yield batch

        batch_size = 2
        num_batches = 10

        vector_generator = random_vector_generator(batch_size * num_batches)
        ids = [str(i) for i in range(batch_size * num_batches)]
        vectors = [t.flatten() for t in next(vector_generator)]

        for i in range(num_batches):
            chromadb_instance.insert(
                ids=ids[batch_size * i : batch_size * (i + 1)],
                vectors=vectors[batch_size * i : batch_size * (i + 1)],
            )

        ids = ["id1", "id2"]
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        chromadb_instance.insert(ids, vectors)

        # Search should return 1 id when k is 1
        result = chromadb_instance.search_by_vector([np.array([1, 2, 3])], k=1)
        assert len(result[0]) == 1
        assert np.all(result[0] == ["id1"])


def test_get_by_id():
    """Test get_by_id retrieves embeddings"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chromadb_instance = ChromaDB({}, {"results_dir": temp_dir})
        chromadb_instance.connect()
        chromadb_instance.create()

        ids = ["id1", "id2"]
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        chromadb_instance.insert(ids, vectors)

        result = chromadb_instance.get_by_id("id1")
        assert np.all(result["id1"] == [1, 2, 3])

        result = chromadb_instance.get_by_id(["id1", "id2"])
        assert len(result) == 2
        assert np.all(result["id1"] == [1, 2, 3])
        assert np.all(result["id2"] == [4, 5, 6])


def test_get_by_id_many_shards():
    """Test get_by_id retrieves embeddings from multiple shards"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chromadb_instance = ChromaDB({}, {"results_dir": temp_dir})
        chromadb_instance.shard_size_limit = 5
        chromadb_instance.min_shards_for_parallelization = 3
        chromadb_instance.connect()
        chromadb_instance.create()

        def random_vector_generator(batch_size=1):
            """Create random vectors"""
            while True:
                batch = [np.random.rand(3) for _ in range(batch_size)]
                yield batch

        batch_size = 2
        num_batches = 10

        vector_generator = random_vector_generator(batch_size * num_batches)
        ids = [str(i) for i in range(batch_size * num_batches)]
        vectors = [t.flatten() for t in next(vector_generator)]

        for i in range(num_batches):
            chromadb_instance.insert(
                ids=ids[batch_size * i : batch_size * (i + 1)],
                vectors=vectors[batch_size * i : batch_size * (i + 1)],
            )

        ids = ["id1", "id2"]
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        chromadb_instance.insert(ids, vectors)

        result = chromadb_instance.get_by_id("id1")
        assert np.all(result["id1"] == [1, 2, 3])
