# TODO: Remove this file once all users have migrated .npy results to Lance format
# and InferenceDataset / InferenceDatasetWriter have been fully deprecated.
# Tracking issue: https://github.com/lincc-frameworks/hyrax/issues/428
"""Tests for scripts/convert_results.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the script module without adding scripts/ to sys.path permanently
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "convert_results.py"
_spec = importlib.util.spec_from_file_location("convert_results", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

convert = _mod.convert
verify = _mod.verify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_npy_fixture(result_dir: Path, batches: list[tuple[list[str], list[np.ndarray]]]) -> None:
    """Write the .npy batch files that InferenceDatasetWriter would produce.

    Parameters
    ----------
    result_dir : Path
        Directory to write the files into.
    batches : list of (ids, tensors) pairs
        ``ids`` is a list of string IDs; ``tensors`` is a list of uniformly-shaped
        numpy arrays.
    """
    id_dtype = np.array([id_ for batch_ids, _ in batches for id_ in batch_ids]).dtype

    all_ids: list[str] = []
    all_batch_nums: list[int] = []

    for batch_num, (ids, tensors) in enumerate(batches):
        first = tensors[0]
        structured_type = np.dtype([("id", id_dtype), ("tensor", first.dtype, first.shape)])
        batch_arr = np.zeros(len(ids), structured_type)
        batch_arr["id"] = ids
        batch_arr["tensor"] = tensors
        np.save(result_dir / f"batch_{batch_num}.npy", batch_arr, allow_pickle=False)

        all_ids.extend(ids)
        all_batch_nums.extend([batch_num] * len(ids))

    # Write batch_index.npy sorted by id (matches InferenceDatasetWriter behaviour)
    index_dtype = np.dtype([("id", id_dtype), ("batch_num", np.int64)])
    batch_index = np.zeros(len(all_ids), index_dtype)
    batch_index["id"] = all_ids
    batch_index["batch_num"] = all_batch_nums
    batch_index.sort(order="id")
    np.save(result_dir / "batch_index.npy", batch_index, allow_pickle=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_npy_dir(tmp_path):
    """Two-batch fixture with 1-D float32 tensors."""
    batches = [
        (["obj_0", "obj_1", "obj_2"], [np.array([i, i + 1], dtype=np.float32) for i in range(3)]),
        (["obj_3", "obj_4"], [np.array([i, i + 1], dtype=np.float32) for i in range(3, 5)]),
    ]
    write_npy_fixture(tmp_path, batches)
    return tmp_path


@pytest.fixture()
def multidim_npy_dir(tmp_path):
    """Single-batch fixture with 2-D float32 tensors of shape (2, 3)."""
    tensors = [np.array([[i, i + 1, i + 2], [i + 3, i + 4, i + 5]], dtype=np.float32) for i in range(4)]
    batches = [(["a", "b", "c", "d"], tensors)]
    write_npy_fixture(tmp_path, batches)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_roundtrip(simple_npy_dir, tmp_path):
    """convert() + verify() succeeds and row count matches."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    convert(simple_npy_dir, output_dir)
    verify(simple_npy_dir, output_dir)

    # Lance DB must exist after conversion
    assert (output_dir / "lance_db").exists()


def test_multidim_tensors_roundtrip(multidim_npy_dir, tmp_path):
    """Multi-dimensional tensors (shape 2×3) survive the round trip."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    convert(multidim_npy_dir, output_dir)

    from hyrax.datasets.result_dataset import ResultDataset

    dataset = ResultDataset({}, output_dir)
    assert len(dataset) == 4
    assert dataset[0].shape == (2, 3)

    verify(multidim_npy_dir, output_dir)


def test_separate_output_dir_leaves_npy_intact(simple_npy_dir, tmp_path):
    """When output_dir differs from input_dir, the .npy files are untouched."""
    output_dir = tmp_path / "lance_output"
    output_dir.mkdir()

    convert(simple_npy_dir, output_dir)

    # Original .npy files must still exist
    assert (simple_npy_dir / "batch_index.npy").exists()
    assert (simple_npy_dir / "batch_0.npy").exists()
    assert (simple_npy_dir / "batch_1.npy").exists()

    # Lance DB must not exist inside input_dir (it went to output_dir)
    assert not (simple_npy_dir / "lance_db").exists()

    verify(simple_npy_dir, output_dir)


def test_inplace_conversion(simple_npy_dir):
    """When input_dir == output_dir, Lance files appear alongside the .npy files."""
    convert(simple_npy_dir, simple_npy_dir)
    verify(simple_npy_dir, simple_npy_dir)

    assert (simple_npy_dir / "lance_db").exists()
    assert (simple_npy_dir / "batch_0.npy").exists()


def test_already_converted_raises(simple_npy_dir, tmp_path):
    """Calling convert() twice on the same output_dir raises RuntimeError."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    convert(simple_npy_dir, output_dir)

    with pytest.raises(RuntimeError, match="already exists"):
        convert(simple_npy_dir, output_dir)


def test_missing_batch_index_raises(tmp_path):
    """convert() raises RuntimeError when batch_index.npy is missing."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(RuntimeError, match="batch_index.npy"):
        convert(empty_dir, tmp_path / "output")


def test_full_bitwise_verification(tmp_path):
    """Every row's ID and tensor matches exactly between numpy source and Lance output."""
    rng = np.random.default_rng(42)
    n_batches = 4
    batch_size = 8
    tensor_shape = (5,)

    batches = []
    for b in range(n_batches):
        ids = [f"obj_{b * batch_size + i:04d}" for i in range(batch_size)]
        tensors = [rng.random(tensor_shape).astype(np.float32) for _ in range(batch_size)]
        batches.append((ids, tensors))

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    write_npy_fixture(input_dir, batches)
    convert(input_dir, output_dir)

    # verify() checks every single row for bitwise equality
    verify(input_dir, output_dir)
