from unittest.mock import MagicMock

import lancedb
import pyarrow as pa
import pytest

from hyrax.datasets.lancedb_dataset import LanceDBDataset


@pytest.fixture
def lancedb_fixture(tmp_path):
    """Create a real LanceDB database on disk with a single 'observations' table
    containing three rows across three columns (object_id, flux, label).

    Used by tests that need to verify behaviour against an actual LanceDB file
    rather than a mock.
    """
    db_path = tmp_path / "demo_lancedb"
    db = lancedb.connect(str(db_path))

    table_data = pa.table(
        {
            "object_id": ["obj_1", "obj_2", "obj_3"],
            "flux": [10.5, 20.5, 30.5],
            "label": [0, 1, 0],
        }
    )
    db.create_table("observations", table_data)
    return db_path


def test_lancedb_dataset_reads_table_and_exposes_getters(lancedb_fixture):
    """Verify that LanceDBDataset opens a real LanceDB table and dynamically creates
    ``get_<field>`` accessor methods for every column.

    Checks that the dataset length matches the number of rows in the table and that
    each accessor returns the correct scalar value for a given row index.
    """
    dataset = LanceDBDataset(
        config={
            "data_set": {
                "LanceDBDataset": {
                    "table_name": False,
                    "connect_kwargs": {},
                    "open_table_kwargs": {},
                }
            }
        },
        data_location=lancedb_fixture,
    )

    assert len(dataset) == 3
    assert dataset.get_object_id(0) == "obj_1"
    assert dataset.get_flux(1) == 20.5
    assert dataset.get_label(2) == 0


def test_lancedb_dataset_passes_through_connect_and_open_table_kwargs(lancedb_fixture, monkeypatch):
    """Verify that ``connect_kwargs`` and ``open_table_kwargs`` from the config are
    forwarded to ``lancedb.connect`` and ``db.open_table`` respectively.

    Uses a wrapper around ``lancedb.connect`` to capture the keyword arguments it
    received, and a mock table to inspect how ``open_table`` was called.  This
    ensures remote/cloud LanceDB connections (which require auth or storage options)
    work correctly when configured via Hyrax config.
    """
    original_connect = lancedb.connect
    mock_db = MagicMock()

    def wrapped_connect(uri, **kwargs):
        wrapped_connect.kwargs = kwargs
        _ = original_connect(uri)
        return mock_db

    wrapped_connect.kwargs = None
    mock_table = MagicMock()
    mock_table.schema.names = ["object_id"]
    mock_table.to_lance.return_value = MagicMock()
    mock_table.count_rows.return_value = 0
    mock_db.open_table.return_value = mock_table

    monkeypatch.setattr(lancedb, "connect", wrapped_connect)

    dataset = LanceDBDataset(
        config={
            "data_set": {
                "LanceDBDataset": {
                    "table_name": "observations",
                    "connect_kwargs": {"api_key": "example-key"},
                    "open_table_kwargs": {"storage_options": {"region": "us-west-2"}},
                }
            }
        },
        data_location=lancedb_fixture,
    )

    assert wrapped_connect.kwargs == {"api_key": "example-key"}
    mock_db.open_table.assert_called_once_with("observations", storage_options={"region": "us-west-2"})
    assert len(dataset) == 0


def test_lancedb_dataset_requires_data_location():
    """Verify that constructing a LanceDBDataset without a ``data_location`` raises
    a ``ValueError`` with a clear error message.

    ``data_location`` points to the LanceDB database directory and has no
    meaningful default, so its absence should be caught eagerly before any
    connection attempt is made.
    """
    with pytest.raises(ValueError):
        LanceDBDataset(config={"data_set": {"LanceDBDataset": {}}})


def test_lancedb_dataset_infers_only_table_name(monkeypatch, tmp_path):
    """Verify that when ``table_name`` is unset (``False``) and the database contains
    exactly one table, that table is automatically selected without requiring explicit
    configuration.

    Uses a mocked LanceDB connection so the test does not require a real database
    on disk, and asserts that ``open_table`` is called with the inferred table name.
    """
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_table.schema.names = ["object_id"]
    mock_table.to_lance.return_value = MagicMock()
    mock_table.count_rows.return_value = 0
    mock_db.table_names.return_value = ["only_table"]
    mock_db.open_table.return_value = mock_table
    monkeypatch.setattr(lancedb, "connect", lambda *_args, **_kwargs: mock_db)

    dataset = LanceDBDataset(
        config={
            "data_set": {
                "LanceDBDataset": {
                    "table_name": False,
                    "connect_kwargs": {},
                    "open_table_kwargs": {},
                }
            }
        },
        data_location=tmp_path / "db",
    )

    mock_db.open_table.assert_called_once_with("only_table")
    assert len(dataset) == 0


def test_lancedb_dataset_requires_table_name_when_multiple_tables(monkeypatch, tmp_path):
    """Verify that when ``table_name`` is unset (``False``) and the database contains
    multiple tables, a ``RuntimeError`` is raised listing the available table names.

    Automatic table inference is only safe when there is exactly one table.  With
    multiple tables the user must specify which one to open, and the error message
    should include the names of the existing tables to help the user fix their config.
    """
    mock_db = MagicMock()
    mock_db.table_names.return_value = ["table_a", "table_b"]
    monkeypatch.setattr(lancedb, "connect", lambda *_args, **_kwargs: mock_db)

    with pytest.raises(RuntimeError, match="table_a, table_b"):
        LanceDBDataset(
            config={
                "data_set": {
                    "LanceDBDataset": {
                        "table_name": False,
                        "connect_kwargs": {},
                        "open_table_kwargs": {},
                    }
                }
            },
            data_location=tmp_path / "db",
        )
