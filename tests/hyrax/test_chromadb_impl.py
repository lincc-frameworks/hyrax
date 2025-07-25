import numpy as np
import pytest

from hyrax import Hyrax
from hyrax.vector_dbs.chromadb_impl import ChromaDB


@pytest.fixture()
def random_vector_generator(batch_size=1, vector_size=3):
    """Create random vectors"""

    def _generator(batch_size=1, vector_size=3):
        while True:
            batch = [np.random.rand(vector_size) for _ in range(batch_size)]
            yield batch

    return _generator


@pytest.fixture()
def chromadb_instance(tmp_path):
    """Create a ChromaDB instance for testing"""
    h = Hyrax()
    chromadb_instance = ChromaDB(h.config, {"results_dir": tmp_path})
    chromadb_instance.connect()
    chromadb_instance.create()
    return chromadb_instance


def test_connect(tmp_path):
    """Test that we can create a connections to the database"""
    h = Hyrax()
    chromadb_instance = ChromaDB(h.config, {"results_dir": tmp_path})
    chromadb_instance.connect()

    assert chromadb_instance.chromadb_client is not None


def test_create(chromadb_instance):
    """Test creation of a single collection (shard) in the database"""
    collections = chromadb_instance.chromadb_client.list_collections()

    assert collections is not None
    assert len(collections) == 1
    assert collections[0].name == "shard_0"


def test_insert(chromadb_instance):
    """Ensure that we can insert IDs and vectors into the database"""

    ids = ["id1", "id2"]
    vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    chromadb_instance.insert(ids, vectors)
    collection = chromadb_instance.chromadb_client.get_collection("shard_0")
    assert collection.count() == 2


@pytest.mark.slow
def test_insert_creates_new_shards(caplog, chromadb_instance, random_vector_generator):
    """Ensure that we can insert IDs and vectors into the database, and that new
    shards are created when the shard size limit is reached"""

    chromadb_instance.shard_size_limit = 5
    chromadb_instance.min_shards_for_parallelization = 3

    batch_size = 2
    num_batches = 10

    vector_generator = random_vector_generator(batch_size * num_batches)
    ids = [str(i) for i in range(batch_size * num_batches)]
    vectors = [t.flatten() for t in next(vector_generator)]

    for i in range(num_batches):
        with caplog.at_level("WARNING"):
            chromadb_instance.insert(
                ids=ids[batch_size * i : batch_size * (i + 1)],
                vectors=vectors[batch_size * i : batch_size * (i + 1)],
            )
        assert "poor performance" not in caplog.text

    collections = chromadb_instance.chromadb_client.list_collections()
    assert len(collections) == 5


def test_insert_raises_warning(caplog, chromadb_instance, random_vector_generator):
    """Ensure that inserting a single large vector logs a warning"""
    batch_size = 1
    vector_size = 10_000

    vector_generator = random_vector_generator(batch_size, vector_size=vector_size)
    ids = [str(i) for i in range(batch_size)]
    vectors = [t.flatten() for t in next(vector_generator)]

    with caplog.at_level("WARNING"):
        chromadb_instance.insert(ids=ids, vectors=vectors)

    assert "poor performance" in caplog.text


def test_insert_does_not_raise_warning(caplog, tmp_path, random_vector_generator):
    """Ensure that inserting a single large vector does not log a warning
    when the config vector_size_warning is set to `False`."""

    h = Hyrax()
    h.config["vector_db"]["chromadb"]["vector_size_warning"] = False
    chromadb_instance = ChromaDB(h.config, {"results_dir": tmp_path})
    chromadb_instance.connect()
    chromadb_instance.create()

    batch_size = 1
    vector_size = 10_000

    vector_generator = random_vector_generator(batch_size, vector_size=vector_size)
    ids = [str(i) for i in range(batch_size)]
    vectors = [t.flatten() for t in next(vector_generator)]

    with caplog.at_level("WARNING"):
        chromadb_instance.insert(ids=ids, vectors=vectors)

    assert "poor performance" not in caplog.text


def test_search_by_id(chromadb_instance):
    """Test search_by_id retrieves nearest neighbor ids"""

    ids = ["id1", "id2"]
    vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    chromadb_instance.insert(ids, vectors)

    # Search by single vector should return the id1 and id2 in that order
    result = chromadb_instance.search_by_id("id1", k=2)
    assert len(result["id1"]) == 2
    assert np.all(result["id1"] == ["id1", "id2"])

    # Search should return all ids when k is larger than the number of ids
    result = chromadb_instance.search_by_id("id1", k=5)
    assert len(result["id1"]) == 2
    assert np.all(result["id1"] == ["id1", "id2"])

    # Search should return 1 id when k is 1
    result = chromadb_instance.search_by_id("id1", k=1)
    assert len(result["id1"]) == 1
    assert np.all(result["id1"] == ["id1"])

    # Search by another vector should return the id2 and id1 in that order
    result = chromadb_instance.search_by_id("id2", k=2)
    assert len(result["id2"]) == 2
    assert np.all(result["id2"] == ["id2", "id1"])


@pytest.mark.slow
def test_search_by_id_many_shards(chromadb_instance, random_vector_generator):
    """Test search_by_id retrieves nearest neighbor ids when there are many shards"""

    chromadb_instance.shard_size_limit = 5
    chromadb_instance.min_shards_for_parallelization = 3

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
    assert len(result["id1"]) == 1
    assert np.all(result["id1"] == ["id1"])


def test_search_by_vector(chromadb_instance):
    """Test search_by_vector retrieves nearest neighbor ids"""

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

    # Search by multiple vectors should return the ids in the order of the vectors
    result = chromadb_instance.search_by_vector([np.array([4, 5, 6]), np.array([1, 2, 3])], k=2)
    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 2
    assert np.all(result[0] == ["id2", "id1"])
    assert np.all(result[1] == ["id1", "id2"])


def test_search_by_vector_not_list(chromadb_instance):
    """Test search_by_vector retrieves nearest neighbor ids when a single vector
    is provided. i.e. not a list of vectors."""

    ids = ["id1", "id2"]
    vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    chromadb_instance.insert(ids, vectors)

    # Search by a single vector should return the id2 and id1 in that order
    result = chromadb_instance.search_by_vector(np.array([4, 5, 6]), k=2)
    assert len(result[0]) == 2
    assert np.all(result[0] == ["id2", "id1"])

    # Search by a non-np.array vector should return the id2 and id1 in that order
    result = chromadb_instance.search_by_vector([4, 5, 6], k=2)
    assert len(result[0]) == 2
    assert np.all(result[0] == ["id2", "id1"])


@pytest.mark.slow
def test_search_by_vector_many_shards(chromadb_instance, random_vector_generator):
    """Test search_by_vector retrieves nearest neighbor ids when there are many shards"""

    chromadb_instance.shard_size_limit = 5
    chromadb_instance.min_shards_for_parallelization = 3

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


def test_get_by_id(chromadb_instance):
    """Test get_by_id retrieves embeddings"""

    ids = ["id1", "id2"]
    vectors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    chromadb_instance.insert(ids, vectors)

    result = chromadb_instance.get_by_id("id1")
    assert np.all(result["id1"] == [1, 2, 3])

    result = chromadb_instance.get_by_id(["id1", "id2"])
    assert len(result) == 2
    assert np.all(result["id1"] == [1, 2, 3])
    assert np.all(result["id2"] == [4, 5, 6])


@pytest.mark.slow
def test_get_by_id_many_shards(chromadb_instance, random_vector_generator):
    """Test get_by_id retrieves embeddings from multiple shards"""

    chromadb_instance.shard_size_limit = 5
    chromadb_instance.min_shards_for_parallelization = 3

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

    for indx, id in enumerate(ids):
        result = chromadb_instance.get_by_id(id)
        assert np.all(result[id] == vectors[indx])
