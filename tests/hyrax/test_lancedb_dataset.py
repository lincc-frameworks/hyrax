from unittest.mock import MagicMock

import lancedb
import pyarrow as pa
import pytest

from hyrax.datasets.lancedb_dataset import LanceDBDataset


@pytest.fixture
def lancedb_fixture(tmp_path):
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


def test_lancedb_dataset_passes_through_connect_and_open_table_kwargs(
    lancedb_fixture, monkeypatch
):
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
    mock_db.open_table.assert_called_once_with(
        "observations", storage_options={"region": "us-west-2"}
    )
    assert len(dataset) == 0


def test_lancedb_dataset_requires_data_location():
    with pytest.raises(ValueError):
        LanceDBDataset(config={"data_set": {"LanceDBDataset": {}}})


def test_lancedb_dataset_infers_only_table_name(monkeypatch, tmp_path):
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
