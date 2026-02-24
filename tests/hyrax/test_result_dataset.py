"""Tests for Lance-based ResultDataset and ResultDatasetWriter."""

import numpy as np
import pytest

import hyrax
from hyrax.data_sets.result_dataset import ResultDataset, ResultDatasetWriter


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    object_ids = np.array(["obj_0", "obj_1", "obj_2", "obj_3", "obj_4"])
    data = [np.array([i, i + 1], dtype=np.float32) for i in range(5)]
    return object_ids, data


@pytest.fixture
def multidim_data():
    """Create multi-dimensional sample data."""
    object_ids = np.array(["obj_0", "obj_1", "obj_2"])
    data = [np.array([[i, i + 1], [i + 2, i + 3]], dtype=np.float32) for i in range(3)]
    return object_ids, data


def test_writer_basic(tmp_path, sample_data):
    """Test basic write operation."""
    object_ids, data = sample_data

    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    # Verify lance_db directory was created
    lance_dir = tmp_path / "lance_db"
    assert lance_dir.exists()


def test_writer_multiple_batches(tmp_path):
    """Test writing multiple batches incrementally."""
    writer = ResultDatasetWriter(tmp_path)

    # Write first batch
    batch1_ids = np.array(["obj_0", "obj_1", "obj_2"])
    batch1_data = [np.array([i, i + 1], dtype=np.float32) for i in range(3)]
    writer.write_batch(batch1_ids, batch1_data)

    # Write second batch
    batch2_ids = np.array(["obj_3", "obj_4"])
    batch2_data = [np.array([i, i + 1], dtype=np.float32) for i in range(3, 5)]
    writer.write_batch(batch2_ids, batch2_data)

    writer.commit()

    # Verify total records
    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)
    assert len(dataset) == 5


def test_writer_multidim_tensors(tmp_path, multidim_data):
    """Test writing multi-dimensional tensors."""
    object_ids, data = multidim_data

    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    # Verify data can be read back
    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)
    assert len(dataset) == 3

    # Check shape is preserved
    read_data = dataset[0]
    assert read_data.shape == (2, 2)


def test_writer_different_dtypes(tmp_path):
    """Test writing tensors with different dtypes."""
    dtypes = [np.float32, np.float64, np.int32, np.int64]

    for dtype in dtypes:
        dtype_name = np.dtype(dtype).name
        result_dir = tmp_path / f"test_{dtype_name}"
        result_dir.mkdir()

        writer = ResultDatasetWriter(result_dir)
        object_ids = np.array(["obj_0", "obj_1"])
        data = [np.array([1.5, 2.5], dtype=dtype), np.array([3.5, 4.5], dtype=dtype)]
        writer.write_batch(object_ids, data)
        writer.commit()

        # Verify dtype is preserved
        h = hyrax.Hyrax()
        dataset = ResultDataset(h.config, result_dir)
        read_data = dataset[0]
        assert read_data.dtype == dtype


def test_reader_basic(tmp_path, sample_data):
    """Test basic read operation."""
    object_ids, data = sample_data

    # Write data
    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    # Read data
    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)
    assert len(dataset) == 5

    # Test single index access
    item = dataset[0]
    assert isinstance(item, np.ndarray)
    assert item.shape == (2,)
    np.testing.assert_array_equal(item, data[0])


def test_reader_getitem_single_index(tmp_path, sample_data):
    """Test __getitem__ with single index."""
    object_ids, data = sample_data

    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)

    # Test each index
    for i in range(5):
        item = dataset[i]
        np.testing.assert_array_equal(item, data[i])


def test_reader_getitem_array_index(tmp_path, sample_data):
    """Test __getitem__ with array of indices."""
    object_ids, data = sample_data

    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)

    # Test array indexing
    indices = np.array([0, 2, 4])
    items = dataset[indices]
    assert items.shape == (3, 2)
    for i, idx in enumerate(indices):
        np.testing.assert_array_equal(items[i], data[idx])


def test_reader_getitem_out_of_range(tmp_path, sample_data):
    """Test __getitem__ raises IndexError for out of range indices."""
    object_ids, data = sample_data

    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)

    # Test out of range index raises IndexError
    with pytest.raises(IndexError):
        _ = dataset[10]

    with pytest.raises(IndexError):
        _ = dataset[-1]


def test_reader_get_object_id(tmp_path, sample_data):
    """Test get_object_id HyraxQL getter."""
    object_ids, data = sample_data

    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)

    # Test each object ID
    for i, expected_id in enumerate(object_ids):
        actual_id = dataset.get_object_id(i)
        assert actual_id == expected_id


def test_reader_get_data(tmp_path, sample_data):
    """Test get_data HyraxQL getter."""
    object_ids, data = sample_data

    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)

    # Test get_data returns same as __getitem__
    for i in range(len(data)):
        get_data_result = dataset.get_data(i)
        getitem_result = dataset[i]
        np.testing.assert_array_equal(get_data_result, getitem_result)


def test_reader_ids(tmp_path, sample_data):
    """Test ids() generator."""
    object_ids, data = sample_data

    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)

    # Test ids generator
    ids_list = list(dataset.ids())
    assert len(ids_list) == len(object_ids)
    for expected_id, actual_id in zip(object_ids, ids_list):
        assert expected_id == actual_id


def test_roundtrip_fidelity(tmp_path):
    """Test that data written and read back is bitwise identical."""
    # Create test data with various edge cases
    object_ids = np.array(["obj_0", "obj_1", "obj_2", "obj_3"])
    data = [
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([np.nan, np.inf], dtype=np.float32),
        np.array([-np.inf, -0.0], dtype=np.float32),
        np.array([1e-45, 1e38], dtype=np.float32),  # denormal and large values
    ]

    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)

    # Verify bitwise equality
    for i, expected in enumerate(data):
        actual = dataset[i]
        # Use array_equal which handles NaN correctly
        np.testing.assert_array_equal(actual, expected)


def test_empty_batch(tmp_path):
    """Test that empty batches are handled gracefully."""
    writer = ResultDatasetWriter(tmp_path)

    # Write empty batch (should be no-op)
    writer.write_batch(np.array([]), [])

    # Write actual data
    writer.write_batch(np.array(["obj_0"]), [np.array([1.0, 2.0], dtype=np.float32)])
    writer.commit()

    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)
    assert len(dataset) == 1


def test_writer_mismatched_lengths(tmp_path):
    """Test that writer raises error for mismatched ID and data lengths."""
    writer = ResultDatasetWriter(tmp_path)

    with pytest.raises(ValueError):
        writer.write_batch(
            np.array(["obj_0", "obj_1"]),  # 2 IDs
            [np.array([1.0, 2.0], dtype=np.float32)],  # 1 data item
        )


def test_reader_nonexistent_directory(tmp_path):
    """Test that reader raises error for non-existent directory."""
    h = hyrax.Hyrax()
    nonexistent_dir = tmp_path / "does_not_exist"

    with pytest.raises(RuntimeError, match="does not exist"):
        ResultDataset(h.config, nonexistent_dir)


def test_iteration(tmp_path, sample_data):
    """Test that dataset can be iterated."""
    object_ids, data = sample_data

    writer = ResultDatasetWriter(tmp_path)
    writer.write_batch(object_ids, data)
    writer.commit()

    h = hyrax.Hyrax()
    dataset = ResultDataset(h.config, tmp_path)

    # Test iteration
    count = 0
    for i, item in enumerate(dataset):
        np.testing.assert_array_equal(item, data[i])
        count += 1

    assert count == len(data)
