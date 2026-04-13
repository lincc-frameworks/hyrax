# TODO: Remove this file once all users have migrated .npy results to Lance format
# and InferenceDataset / InferenceDatasetWriter have been fully deprecated.
# Tracking issue: https://github.com/lincc-frameworks/hyrax/issues/428
"""Convert hyrax inference results from legacy .npy format to Lance format.

Usage
-----
    python scripts/convert_results.py \\
        --input-dir  ./results/infer_20250201_120000 \\
        --output-dir ./results/infer_20250201_120000_lance

``--input-dir`` and ``--output-dir`` may be the same path for an in-place
conversion.  When they differ the original ``.npy`` files are left untouched;
delete them manually after verifying the output.

The script will refuse to overwrite an existing ``lance_db/`` directory inside
``--output-dir``.  Delete it first if you need to re-run the conversion.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def convert(input_dir: Path, output_dir: Path) -> None:
    """Convert a .npy result directory to Lance format.

    Parameters
    ----------
    input_dir : Path
        Directory that contains ``batch_index.npy`` and ``batch_*.npy`` files
        written by ``InferenceDatasetWriter``.
    output_dir : Path
        Directory where the Lance database will be written.  The Lance files
        are placed under ``output_dir/lance_db/``.  ``output_dir`` is created
        if it does not exist.

    Raises
    ------
    RuntimeError
        If ``batch_index.npy`` is missing from *input_dir*, or if
        ``output_dir/lance_db/`` already exists.
    """
    from hyrax.datasets.result_dataset import ResultDatasetWriter

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Validate input
    batch_index_path = input_dir / "batch_index.npy"
    if not batch_index_path.exists():
        raise RuntimeError(
            f"No batch_index.npy found in {input_dir}. Is this a valid .npy inference result directory?"
        )

    # Guard against overwriting existing Lance data
    lance_dir = output_dir / "lance_db"
    if lance_dir.exists():
        raise RuntimeError(
            f"{lance_dir} already exists. Delete it first if you want to re-run the conversion."
        )

    # Load index to know the expected total row count (used later in verify)
    batch_index = np.load(batch_index_path)
    total_expected = len(batch_index)
    print(f"Found {total_expected} records across batch_index.npy")

    # Collect and sort batch files numerically
    def _batch_num(p: Path) -> int:
        m = re.fullmatch(r"batch_(\d+)\.npy", p.name)
        return int(m.group(1)) if m else -1

    batch_files = sorted(
        [p for p in input_dir.glob("batch_*.npy") if re.fullmatch(r"batch_\d+\.npy", p.name)],
        key=_batch_num,
    )

    if not batch_files:
        raise RuntimeError(f"No batch_*.npy files found in {input_dir}.")

    print(f"Converting {len(batch_files)} batch file(s) to Lance …")

    writer = ResultDatasetWriter(output_dir)
    total_written = 0

    for batch_path in batch_files:
        batch_data = np.load(batch_path)
        ids = batch_data["id"].astype(str)
        tensors = list(batch_data["tensor"])
        writer.write_batch(ids, tensors)
        total_written += len(ids)
        print(f"  {batch_path.name}: {len(ids)} records")

    writer.commit()
    print(f"Conversion complete — {total_written} records written to {output_dir / 'lance_db'}")


def verify(input_dir: Path, output_dir: Path) -> None:
    """Verify every record in the Lance output matches the original .npy source.

    Reads all batch files from *input_dir* into an in-memory lookup table, then
    iterates every row of the Lance dataset and asserts bitwise equality for
    both the object ID and the tensor.

    Parameters
    ----------
    input_dir : Path
        Directory containing the original ``batch_*.npy`` files.
    output_dir : Path
        Directory containing the converted ``lance_db/`` Lance database.

    Raises
    ------
    RuntimeError
        If row counts differ, if an ID is missing from the numpy source, or if
        any tensor value does not match exactly.
    """
    from hyrax.datasets.result_dataset import ResultDataset

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Build id -> tensor lookup from all batch files
    def _batch_num(p: Path) -> int:
        m = re.fullmatch(r"batch_(\d+)\.npy", p.name)
        return int(m.group(1)) if m else -1

    batch_files = sorted(
        [p for p in input_dir.glob("batch_*.npy") if re.fullmatch(r"batch_\d+\.npy", p.name)],
        key=_batch_num,
    )

    id_to_tensor: dict[str, np.ndarray] = {}
    for batch_path in batch_files:
        batch_data = np.load(batch_path)
        for row in batch_data:
            id_to_tensor[str(row["id"])] = row["tensor"]

    # Row-count check against batch_index.npy
    batch_index = np.load(input_dir / "batch_index.npy")
    expected_count = len(batch_index)

    dataset = ResultDataset({}, output_dir)
    actual_count = len(dataset)

    if actual_count != expected_count:
        raise RuntimeError(
            f"Row count mismatch: Lance has {actual_count} rows, "
            f"but batch_index.npy has {expected_count} rows."
        )

    # Verify every row bitwise
    print(f"Verifying {actual_count} records …")
    for i in range(actual_count):
        obj_id = dataset.get_object_id(i)
        if obj_id not in id_to_tensor:
            raise RuntimeError(f"Row {i}: object_id {obj_id!r} found in Lance but not in any batch file.")
        lance_tensor = np.asarray(dataset[i])
        numpy_tensor = id_to_tensor[obj_id]
        if not np.array_equal(lance_tensor, numpy_tensor):
            raise RuntimeError(
                f"Row {i} (id={obj_id!r}): tensor mismatch.\n"
                f"  Lance : {lance_tensor}\n"
                f"  NumPy : {numpy_tensor}"
            )

    print(f"Verification passed — all {actual_count} records match.")


def main() -> None:
    """Parse CLI arguments and run the conversion and verification steps."""
    parser = argparse.ArgumentParser(
        description="Convert hyrax .npy inference results to Lance format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="Directory containing batch_index.npy and batch_*.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory where the Lance database will be written.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        convert(input_dir, output_dir)
        verify(input_dir, output_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
