# TODO: Remove this file once all users have migrated .npy results to Lance format
# and InferenceDataset / InferenceDatasetWriter have been fully deprecated.
# Tracking issue: https://github.com/lincc-frameworks/hyrax/issues/428
"""Tests for scripts/convert_results.py."""

import importlib.util
import sys
import unittest.mock
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
    from hyrax.datasets.inference_dataset import InferenceDatasetWriter

    all_ids = [id_ for batch_ids, _ in batches for id_ in batch_ids]

    class _MinimalDataset:
        config: dict = {}

        def ids(self):
            return all_ids

    writer = InferenceDatasetWriter(_MinimalDataset(), result_dir)
    for ids, tensors in batches:
        writer.write_batch(ids, tensors)
    writer.write_index()


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


def test_missing_batch_file_raises(tmp_path):
    """convert() raises RuntimeError when a batch file is missing, leaving row count short."""
    batches = [
        (["a", "b"], [np.array([1.0, 2.0], dtype=np.float32)] * 2),
        (["c", "d"], [np.array([3.0, 4.0], dtype=np.float32)] * 2),
    ]
    write_npy_fixture(tmp_path, batches)

    # Remove one batch file so total_written < total_expected
    (tmp_path / "batch_1.npy").unlink()

    with pytest.raises(RuntimeError, match="Row count mismatch after conversion"):
        convert(tmp_path, tmp_path / "output")


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_full_bitwise_verification(tmp_path, float_dtype):
    """Every row's ID and tensor matches exactly between numpy source and Lance output."""
    rng = np.random.default_rng(42)
    n_batches = 4
    batch_size = 8
    tensor_shape = (5,)

    batches = []
    for b in range(n_batches):
        ids = [f"obj_{b * batch_size + i:04d}" for i in range(batch_size)]
        tensors = [rng.random(tensor_shape).astype(float_dtype) for _ in range(batch_size)]
        batches.append((ids, tensors))

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    write_npy_fixture(input_dir, batches)
    convert(input_dir, output_dir)

    # verify() checks every single row for bitwise equality
    verify(input_dir, output_dir)


def test_skip_verify_flag(simple_npy_dir, tmp_path):
    """--skip-verify causes main() to skip the verify step."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    argv = [
        "convert_results.py",
        "--input-dir",
        str(simple_npy_dir),
        "--output-dir",
        str(output_dir),
        "--skip-verify",
    ]
    with (
        unittest.mock.patch.object(_mod, "verify", wraps=_mod.verify) as mock_verify,
        unittest.mock.patch.object(sys, "argv", argv),
    ):
        _mod.main()

    mock_verify.assert_not_called()
    assert (output_dir / "lance_db").exists()


def test_streaming_verify_many_batches(tmp_path):
    """verify() processes one batch at a time; correct results across many batches."""
    rng = np.random.default_rng(7)
    n_batches = 6
    batch_size = 5

    batches = [
        (
            [f"id_{b * batch_size + i:03d}" for i in range(batch_size)],
            [rng.random((3,)).astype(np.float32) for _ in range(batch_size)],
        )
        for b in range(n_batches)
    ]

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    write_npy_fixture(input_dir, batches)
    convert(input_dir, output_dir)

    # Should not raise; exercises streaming path with multiple batch transitions
    verify(input_dir, output_dir)


def test_index_positions_match(tmp_path):
    """Each object_id occupies the same positional index in Lance as in the .npy batch files.

    The converter writes Lance rows in batch-file order (batch_0, batch_1, …), so
    Lance row i must hold the same object_id as position i in the flattened
    sequence of batch files (not the sorted batch_index.npy).

    Uses 12 batches to verify that batch files are sorted numerically
    (batch_9 before batch_10) rather than lexicographically (batch_10 before batch_9).
    """
    import re

    from hyrax.datasets.result_dataset import ResultDataset

    n_batches = 12
    # Deliberately unsorted IDs within each batch so batch-order != alphabetical order.
    batches = [
        ([f"z{b:02d}", f"a{b:02d}", f"m{b:02d}"], [np.array([float(b)], dtype=np.float32)] * 3)
        for b in range(n_batches)
    ]

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    write_npy_fixture(input_dir, batches)
    convert(input_dir, output_dir)

    # Build expected insertion-order ID list from the batch files sorted numerically.
    def _batch_num(p):
        m = re.fullmatch(r"batch_(\d+)\.npy", p.name)
        return int(m.group(1)) if m else -1

    batch_files = sorted(
        [p for p in input_dir.glob("batch_*.npy") if re.fullmatch(r"batch_\d+\.npy", p.name)],
        key=_batch_num,
    )
    expected_ids = [str(row["id"]) for bf in batch_files for row in np.load(bf)]

    dataset = ResultDataset({}, output_dir)
    actual_ids = [dataset.get_object_id(i) for i in range(len(dataset))]

    assert actual_ids == expected_ids
