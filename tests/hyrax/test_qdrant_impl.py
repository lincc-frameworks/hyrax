import numpy as np
import pytest

from hyrax import Hyrax
from hyrax.vector_dbs.qdrantdb_impl import QdrantDB


@pytest.fixture()
def random_vector_generator(batch_size=1, vector_size=3):
    """Create random vectors"""

    def _generator(batch_size=1, vector_size=3):
        while True:
            batch = [np.random.rand(vector_size) for _ in range(batch_size)]
            yield batch

    return _generator


@pytest.fixture()
def qdrant_instance(tmp_path):
    """Create a QdrantDB instance for testing"""
    h = Hyrax()
    h.config["vector_db.qdrant"]["vector_size"] = 3
    qdrant_instance = QdrantDB(h.config, {"results_dir": tmp_path})
    qdrant_instance.connect()
    qdrant_instance.create()
    return qdrant_instance


def test_connect(tmp_path):
    """Test that we can create a connections to the database"""
    h = Hyrax()
    qdrant_instance = QdrantDB(h.config, {"results_dir": tmp_path})
    qdrant_instance.connect()

    #! It would be nice if this was qdrant.client.heartbeat() or something
    assert qdrant_instance.client is not None


def test_create(qdrant_instance):
    """Test creation of a single collection (shard) in the database"""
    collections = qdrant_instance.client.get_collections().collections

    assert collections is not None
    assert len(collections) == 1
    assert collections[0].name == "shard_0"


def test_insert(qdrant_instance, random_vector_generator):
    """Ensure that we can insert IDs and vectors into the database"""

    batch_size = 20
    num_batches = 10
    vector_generator = random_vector_generator(batch_size * num_batches)
    ids = list(range(batch_size * num_batches))
    vectors = [t.flatten() for t in next(vector_generator)]

    qdrant_instance.insert(ids, vectors)
    total_count = qdrant_instance.client.count(collection_name="shard_0", exact=True)
    assert total_count.count == batch_size * num_batches


def test_search_by_id(qdrant_instance):
    """Test search_by_id retrieves nearest neighbor ids"""

    ids = [1, 2]
    vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    qdrant_instance.insert(ids, vectors)

    # Search by single vector should return the id1 and id2 in that order
    result = qdrant_instance.search_by_id(1, k=2)
    assert len(result[1]) == 2
    assert np.all(result[1] == [1, 2])

    # # Search should return all ids when k is larger than the number of ids
    # result = qdrant_instance.search_by_id("id1", k=5)
    # assert len(result["id1"]) == 2
    # assert np.all(result["id1"] == ["id1", "id2"])

    # # Search should return 1 id when k is 1
    # result = qdrant_instance.search_by_id("id1", k=1)
    # assert len(result["id1"]) == 1
    # assert np.all(result["id1"] == ["id1"])

    # # Search by another vector should return the id2 and id1 in that order
    # result = qdrant_instance.search_by_id("id2", k=2)
    # assert len(result["id2"]) == 2
    # assert np.all(result["id2"] == ["id2", "id1"])


def test_search_by_vector(qdrant_instance):
    """Test search_by_vector retrieves nearest neighbor ids"""

    ids = ["id1", "id2"]
    vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    qdrant_instance.insert(ids, vectors)

    # Search by single vector should return the id1 and id2 in that order
    result = qdrant_instance.search_by_vector([np.array([1, 2, 3])], k=2)
    assert len(result[0]) == 2
    assert np.all(result[0] == ["id1", "id2"])

    # Search should return all ids when k is larger than the number of ids
    result = qdrant_instance.search_by_vector([np.array([1, 2, 3])], k=5)
    assert len(result[0]) == 2
    assert np.all(result[0] == ["id1", "id2"])

    # Search should return 1 id when k is 1
    result = qdrant_instance.search_by_vector([np.array([1, 2, 3])], k=1)
    assert len(result[0]) == 1
    assert np.all(result[0] == ["id1"])

    # Search by multiple vectors should return the ids in the order of the vectors
    result = qdrant_instance.search_by_vector([np.array([4, 5, 6]), np.array([1, 2, 3])], k=2)
    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 2
    assert np.all(result[0] == ["id2", "id1"])
    assert np.all(result[1] == ["id1", "id2"])


def test_search_by_vector_not_list(qdrant_instance):
    """Test search_by_vector retrieves nearest neighbor ids when a single vector
    is provided. i.e. not a list of vectors."""

    ids = ["id1", "id2"]
    vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    qdrant_instance.insert(ids, vectors)

    # Search by a single vector should return the id2 and id1 in that order
    result = qdrant_instance.search_by_vector(np.array([4, 5, 6]), k=2)
    assert len(result[0]) == 2
    assert np.all(result[0] == ["id2", "id1"])

    # Search by a non-np.array vector should return the id2 and id1 in that order
    result = qdrant_instance.search_by_vector([4, 5, 6], k=2)
    assert len(result[0]) == 2
    assert np.all(result[0] == ["id2", "id1"])


# def test_search_by_vector_many_shards(chromadb_instance, random_vector_generator):
#     """Test search_by_vector retrieves nearest neighbor ids when there are many shards"""

#     chromadb_instance.shard_size_limit = 5
#     chromadb_instance.min_shards_for_parallelization = 3

#     batch_size = 2
#     num_batches = 10

#     vector_generator = random_vector_generator(batch_size * num_batches)
#     ids = [str(i) for i in range(batch_size * num_batches)]
#     vectors = [t.flatten() for t in next(vector_generator)]

#     for i in range(num_batches):
#         chromadb_instance.insert(
#             ids=ids[batch_size * i : batch_size * (i + 1)],
#             vectors=vectors[batch_size * i : batch_size * (i + 1)],
#         )

#     ids = ["id1", "id2"]
#     vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
#     chromadb_instance.insert(ids, vectors)

#     # Search should return 1 id when k is 1
#     result = chromadb_instance.search_by_vector([np.array([1, 2, 3])], k=1)
#     assert len(result[0]) == 1
#     assert np.all(result[0] == ["id1"])


def test_get_by_id(qdrant_instance):
    """Test get_by_id retrieves embeddings"""

    ids = ["id1", "id2"]
    vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    qdrant_instance.insert(ids, vectors)

    result = qdrant_instance.get_by_id("id1")
    assert np.all(result["id1"] == [1, 2, 3])

    result = qdrant_instance.get_by_id(["id1", "id2"])
    assert len(result) == 2
    assert np.all(result["id1"] == [1, 2, 3])
    assert np.all(result["id2"] == [4, 5, 6])


# def test_get_by_id_many_shards(chromadb_instance, random_vector_generator):
#     """Test get_by_id retrieves embeddings from multiple shards"""

#     chromadb_instance.shard_size_limit = 5
#     chromadb_instance.min_shards_for_parallelization = 3

#     batch_size = 2
#     num_batches = 10

#     vector_generator = random_vector_generator(batch_size * num_batches)
#     ids = [str(i) for i in range(batch_size * num_batches)]
#     vectors = [t.flatten() for t in next(vector_generator)]

#     for i in range(num_batches):
#         chromadb_instance.insert(
#             ids=ids[batch_size * i : batch_size * (i + 1)],
#             vectors=vectors[batch_size * i : batch_size * (i + 1)],
#         )

#     ids = ["id1", "id2"]
#     vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
#     chromadb_instance.insert(ids, vectors)

#     result = chromadb_instance.get_by_id("id1")
#     assert np.all(result["id1"] == [1, 2, 3])

#     for indx, id in enumerate(ids):
#         result = chromadb_instance.get_by_id(id)
#         assert np.all(result[id] == vectors[indx])
