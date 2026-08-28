import numpy as np
import pyarrow as pa
import pytest

from hyrax import Hyrax
from hyrax.vector_dbs.lance_impl import LanceDB

# Vector size used throughout the tests
_VECTOR_SIZE = 8
# Number of pre-populated vectors in the fixture table.
# Must be > 256 to satisfy Lance's PQ training minimum.
_N_VECTORS = 300


def _make_lance_db(tmp_path, n_vectors=_N_VECTORS, vector_size=_VECTOR_SIZE):
    """Create a small Lance results table under *tmp_path*/lance_db/ and return
    the fixture data as (ids, vectors)."""
    import lancedb

    lance_dir = tmp_path / "lance_db"
    db = lancedb.connect(str(lance_dir))

    ids = [f"id_{i}" for i in range(n_vectors)]
    vectors = np.random.randn(n_vectors, vector_size).astype(np.float32)

    schema = pa.schema(
        [
            pa.field("object_id", pa.string()),
            pa.field("data", pa.list_(pa.float32(), vector_size)),
        ]
    )
    arrow_table = pa.table(
        {
            "object_id": pa.array(ids, type=pa.string()),
            "data": pa.array(vectors.tolist(), type=pa.list_(pa.float32(), vector_size)),
        },
        schema=schema,
    )
    db.create_table("results", arrow_table)
    return ids, vectors


@pytest.fixture()
def lance_instance(tmp_path):
    """Create a LanceDB instance backed by a small pre-populated Lance table."""
    _make_lance_db(tmp_path)
    h = Hyrax()
    # Use tiny partitions so index creation is fast with a small table
    h.config["vector_db"]["lance"]["num_partitions"] = 4
    h.config["vector_db"]["lance"]["num_sub_vectors"] = 2
    instance = LanceDB(h.config, {"results_dir": str(tmp_path)})
    instance.connect()
    instance.create()
    return instance


# ---------------------------------------------------------------------------
# connect / create
# ---------------------------------------------------------------------------


def test_connect(tmp_path):
    """connect() opens the Lance table without raising."""
    _make_lance_db(tmp_path)
    h = Hyrax()
    instance = LanceDB(h.config, {"results_dir": str(tmp_path)})
    instance.connect()
    assert instance.table is not None


def test_create(lance_instance):
    """create() builds an index on the 'data' column."""
    indices = lance_instance.table.list_indices()
    assert len(indices) == 1
    assert "data" in str(indices[0])


def test_create_idempotent(lance_instance):
    """Calling create() twice does not raise (replace=True is the Lance default)."""
    lance_instance.create()  # second call — should succeed silently


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


def test_insert(lance_instance):
    """insert() adds new IDs and vectors to the table."""
    initial_count = lance_instance.table.count_rows()
    new_ids = ["new_id_0", "new_id_1"]
    new_vectors = [np.ones(_VECTOR_SIZE, dtype=np.float32), np.zeros(_VECTOR_SIZE, dtype=np.float32)]
    lance_instance.insert(new_ids, new_vectors)
    assert lance_instance.table.count_rows() == initial_count + 2


def test_insert_deduplicates(lance_instance):
    """insert() skips IDs that already exist in the table."""
    initial_count = lance_instance.table.count_rows()
    existing_id = "id_0"
    lance_instance.insert([existing_id], [np.ones(_VECTOR_SIZE, dtype=np.float32)])
    assert lance_instance.table.count_rows() == initial_count


def test_insert_partial_deduplication(lance_instance):
    """insert() adds only the IDs that do not already exist."""
    initial_count = lance_instance.table.count_rows()
    ids = ["id_0", "brand_new_id"]
    vectors = [np.ones(_VECTOR_SIZE, dtype=np.float32), np.full(_VECTOR_SIZE, 2.0, dtype=np.float32)]
    lance_instance.insert(ids, vectors)
    assert lance_instance.table.count_rows() == initial_count + 1


# ---------------------------------------------------------------------------
# search_by_vector
# ---------------------------------------------------------------------------


def test_search_by_vector_single(lance_instance):
    """search_by_vector returns k results for a single query vector."""
    query = np.random.randn(_VECTOR_SIZE).astype(np.float32)
    result = lance_instance.search_by_vector([query], k=3)
    assert 0 in result
    assert len(result[0]) == 3


def test_search_by_vector_multiple(lance_instance):
    """search_by_vector returns results for each query vector."""
    queries = [np.random.randn(_VECTOR_SIZE).astype(np.float32) for _ in range(3)]
    result = lance_instance.search_by_vector(queries, k=2)
    assert len(result) == 3
    for i in range(3):
        assert len(result[i]) == 2


def test_search_by_vector_1d_ndarray(lance_instance):
    """search_by_vector accepts a plain 1-D ndarray (not wrapped in a list)."""
    query = np.random.randn(_VECTOR_SIZE).astype(np.float32)
    result = lance_instance.search_by_vector(query, k=1)
    assert 0 in result
    assert len(result[0]) == 1


# ---------------------------------------------------------------------------
# search_by_id
# ---------------------------------------------------------------------------


def test_search_by_id(lance_instance):
    """search_by_id returns neighbors keyed by the input ID."""
    result = lance_instance.search_by_id("id_0", k=3)
    assert "id_0" in result
    assert len(result["id_0"]) == 3


def test_search_by_id_nearest_is_self(lance_instance):
    """The closest neighbor of a vector should be itself."""
    result = lance_instance.search_by_id("id_0", k=1)
    assert result["id_0"][0] == "id_0"


def test_search_by_id_missing_raises(lance_instance):
    """search_by_id raises KeyError for an unknown ID."""
    with pytest.raises(KeyError):
        lance_instance.search_by_id("does_not_exist", k=1)


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------


def test_get_by_id(lance_instance):
    """get_by_id returns the vector for a known ID."""
    result = lance_instance.get_by_id(["id_0"])
    assert "id_0" in result
    assert len(result["id_0"]) == _VECTOR_SIZE


def test_get_by_id_multiple(lance_instance):
    """get_by_id returns vectors for all requested IDs."""
    result = lance_instance.get_by_id(["id_0", "id_1", "id_2"])
    assert set(result.keys()) == {"id_0", "id_1", "id_2"}


def test_get_by_id_missing_returns_empty(lance_instance):
    """get_by_id returns an empty dict for IDs that do not exist."""
    result = lance_instance.get_by_id(["ghost_id"])
    assert result == {}
