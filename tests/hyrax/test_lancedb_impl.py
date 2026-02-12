import numpy as np
import pytest

from hyrax import Hyrax
from hyrax.vector_dbs.lancedb_impl import LanceDB


@pytest.fixture()
def random_vector_generator(batch_size=1, vector_size=3):
    """Create random vectors"""

    def _generator(batch_size=1, vector_size=3):
        while True:
            batch = [np.random.rand(vector_size) for _ in range(batch_size)]
            yield batch

    return _generator


@pytest.fixture()
def lancedb_instance(tmp_path):
    """Create a LanceDB instance for testing"""
    h = Hyrax()
    lancedb_instance = LanceDB(h.config, {"results_dir": tmp_path})
    lancedb_instance.connect()
    lancedb_instance.create()
    return lancedb_instance


def test_connect(tmp_path):
    """Test that we can create a connection to the database"""
    h = Hyrax()
    lancedb_instance = LanceDB(h.config, {"results_dir": tmp_path})
    lancedb_instance.connect()

    assert lancedb_instance.db is not None


def test_create(lancedb_instance):
    """Test creation of a table in the database"""
    # Initially table should be None until first insert
    assert lancedb_instance.table is None or lancedb_instance.table is not None


def test_insert(lancedb_instance):
    """Ensure that we can insert IDs and vectors into the database"""
    ids = ["id1", "id2"]
    vectors = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    lancedb_instance.insert(ids, vectors)
    
    # Verify the table was created and vectors were inserted
    assert lancedb_instance.table is not None
    df = lancedb_instance.table.to_pandas()
    assert len(df) == 2


def test_insert_duplicate_ids(lancedb_instance):
    """Ensure that duplicate IDs are not inserted"""
    ids = ["id1", "id2"]
    vectors = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    lancedb_instance.insert(ids, vectors)
    
    # Try to insert the same IDs again
    lancedb_instance.insert(ids, vectors)
    
    # Should still have only 2 rows
    df = lancedb_instance.table.to_pandas()
    assert len(df) == 2


def test_search_by_id(lancedb_instance):
    """Test searching for nearest neighbors by ID"""
    ids = ["id1", "id2", "id3"]
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    lancedb_instance.insert(ids, vectors)
    
    # Search for nearest neighbor of id1
    results = lancedb_instance.search_by_id("id1", k=2)
    
    assert "id1" in results
    assert len(results["id1"]) == 2
    # The first result should be id1 itself
    assert results["id1"][0] == "id1"


def test_search_by_id_not_found(lancedb_instance):
    """Test searching for an ID that doesn't exist"""
    ids = ["id1", "id2"]
    vectors = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    lancedb_instance.insert(ids, vectors)
    
    results = lancedb_instance.search_by_id("id_not_exist", k=1)
    assert results == {}


def test_search_by_id_invalid_k(lancedb_instance):
    """Test that search_by_id raises error for invalid k"""
    with pytest.raises(ValueError, match="k must be greater than 0"):
        lancedb_instance.search_by_id("id1", k=0)


def test_search_by_vector(lancedb_instance):
    """Test searching for nearest neighbors by vector"""
    ids = ["id1", "id2", "id3"]
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    lancedb_instance.insert(ids, vectors)
    
    # Search for nearest neighbors of a vector similar to id1
    query_vector = np.array([0.9, 0.1, 0.0])
    results = lancedb_instance.search_by_vector(query_vector, k=2)
    
    assert 0 in results
    assert len(results[0]) == 2
    # The closest should be id1
    assert results[0][0] == "id1"


def test_search_by_vector_multiple(lancedb_instance):
    """Test searching for nearest neighbors with multiple vectors"""
    ids = ["id1", "id2", "id3"]
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    lancedb_instance.insert(ids, vectors)
    
    # Search with multiple query vectors
    query_vectors = [
        np.array([0.9, 0.1, 0.0]),
        np.array([0.0, 0.9, 0.1]),
    ]
    results = lancedb_instance.search_by_vector(query_vectors, k=1)
    
    assert len(results) == 2
    assert 0 in results
    assert 1 in results


def test_search_by_vector_invalid_k(lancedb_instance):
    """Test that search_by_vector raises error for invalid k"""
    with pytest.raises(ValueError, match="k must be greater than 0"):
        lancedb_instance.search_by_vector(np.array([1.0, 2.0, 3.0]), k=0)


def test_get_by_id(lancedb_instance):
    """Test retrieving vectors by ID"""
    ids = ["id1", "id2", "id3"]
    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]
    lancedb_instance.insert(ids, vectors)
    
    # Get vectors by ID
    results = lancedb_instance.get_by_id(["id1", "id3"])
    
    assert "id1" in results
    assert "id3" in results
    assert len(results) == 2
    
    # Verify the vectors match
    np.testing.assert_array_almost_equal(results["id1"], [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(results["id3"], [7.0, 8.0, 9.0])


def test_get_by_id_not_found(lancedb_instance):
    """Test retrieving vectors for IDs that don't exist"""
    ids = ["id1", "id2"]
    vectors = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    lancedb_instance.insert(ids, vectors)
    
    results = lancedb_instance.get_by_id(["id1", "id_not_exist"])
    
    # Should only return id1
    assert "id1" in results
    assert "id_not_exist" not in results
    assert len(results) == 1


def test_get_by_id_integer_ids(lancedb_instance):
    """Test that integer IDs are handled correctly"""
    ids = [1, 2, 3]
    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]
    lancedb_instance.insert(ids, vectors)
    
    # Get vectors with integer IDs
    results = lancedb_instance.get_by_id([1, 3])
    
    assert 1 in results
    assert 3 in results
    assert len(results) == 2


@pytest.mark.slow
def test_large_batch_insert(lancedb_instance, random_vector_generator):
    """Test inserting a large batch of vectors"""
    batch_size = 1000
    vector_size = 64
    
    vector_generator = random_vector_generator(batch_size, vector_size=vector_size)
    ids = [str(i) for i in range(batch_size)]
    vectors = [t.flatten() for t in next(vector_generator)]
    
    lancedb_instance.insert(ids=ids, vectors=vectors)
    
    # Verify all vectors were inserted
    df = lancedb_instance.table.to_pandas()
    assert len(df) == batch_size


def test_search_empty_database(tmp_path):
    """Test searching in an empty database"""
    h = Hyrax()
    lancedb_instance = LanceDB(h.config, {"results_dir": tmp_path})
    lancedb_instance.connect()
    lancedb_instance.create()
    
    # Search should return empty results
    results = lancedb_instance.search_by_vector(np.array([1.0, 2.0, 3.0]), k=1)
    assert results == {}
