# TODO: Remove this file once all users have migrated .npy results to Lance format
# and InferenceDataset / InferenceDatasetWriter have been fully deprecated.
# Tracking issue: https://github.com/lincc-frameworks/hyrax/issues/428
"""Convert hyrax inference results from legacy .npy format to Lance format.

Usage
-----
    python scripts/convert_results.py \\
        --input-dir  ./results/infer_20250201_120000 \\
        --output-dir ./results/infer_20250201_120000_lance
        [--skip-verify]

``--input-dir`` and ``--output-dir`` may be the same path for an in-place
conversion.  When they differ the original ``.npy`` files are left untouched;
delete them manually after verifying the output.

The script will refuse to overwrite an existing ``lance_db/`` directory inside
``--output-dir``.  Delete it first if you need to re-run the conversion.

Using ``--skip-verify`` is not recommended, but can be used to skip the verification
step after conversion if you are confident in the conversion process and want to
save time.  Note that verification is a crucial step to ensure data integrity,
so use this option with caution.

Notes:
- Using the same --input-dir and --output-dir is fine. The original data WILL NOT
be overwritten or deleted.
- The script processes batch files incrementally to avoid high memory usage, so
it can handle large datasets that do not fit in memory.
- After conversion, by default, the script verifies that every record in the Lance
output matches the original .npy source by checking both object IDs and tensor values.
This behavior can be skipped with the --skip-verify flag, but verification is
recommended to ensure the conversion was successful and data integrity is maintained.
- The original .npy files (batch_*.npy, batch_index.npy, and batch_index_insertion_order.npy)
WILL NOT be deleted after conversion. You must delete them manually if you no
longer need them.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


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

    Notes
    -----
    - The conversion is done batch-by-batch to avoid loading all tensors into
      memory at once. Each batch file is read, converted to Arrow format, and
      appended to the Lance dataset before moving on to the next batch file.
    - The script will print progress messages indicating how many records were
      found in each batch file and the total number of records written to Lance.
    - The original .npy files (batch_*.npy, batch_index.npy, and
      batch_index_insertion_order.npy) WILL NOT be deleted after conversion.
      You must delete them manually if you no longer need them.

    Raises
    ------
    RuntimeError
        If ``batch_index.npy`` is missing from *input_dir*, or if
        ``output_dir/lance_db/`` already exists.
    """
    from hyrax.datasets.result_dataset import LANCE_DB_DIR, ResultDatasetWriter

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Validate input
    batch_index_path = input_dir / "batch_index.npy"
    if not batch_index_path.exists():
        raise RuntimeError(
            f"No batch_index.npy found in {input_dir}. Is this a valid .npy inference result directory?"
        )

    # Guard against overwriting existing Lance data
    lance_dir = output_dir / LANCE_DB_DIR
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

    for batch_path in tqdm(batch_files, desc="Converting", unit="batch"):
        batch_data = np.load(batch_path)
        ids = batch_data["id"].astype(str)
        tensors = list(batch_data["tensor"])
        writer.write_batch(ids, tensors)
        total_written += len(ids)

    writer.commit()

    if total_written != total_expected:
        raise RuntimeError(
            f"Row count mismatch after conversion: wrote {total_written} records "
            f"but expected {total_expected} from batch_index.npy. "
            f"One or more batch files may be missing or corrupt."
        )

    print(f"Conversion complete — {total_written} records written to {output_dir / LANCE_DB_DIR}")


def verify(input_dir: Path, output_dir: Path) -> None:
    """Verify every record in the Lance output matches the original .npy source.

    Processes one numpy batch file at a time, holding only that batch's data in
    memory.  This keeps peak memory proportional to the largest single batch
    rather than the full dataset, making verification feasible on large result
    directories.

    The converter writes Lance rows in the same order as the numpy batches
    (``batch_0``, ``batch_1``, …), so verification can advance a positional
    offset through the Lance table as it steps through each batch file —
    no full in-memory ID lookup table is required.

    Parameters
    ----------
    input_dir : Path
        Directory containing the original ``batch_*.npy`` files.
    output_dir : Path
        Directory containing the converted ``lance_db/`` Lance database.

    Raises
    ------
    RuntimeError
        If row counts differ, if an ID at a given position does not match
        between the numpy source and Lance, or if any tensor value does not
        match exactly.
    """
    from hyrax.datasets.result_dataset import ResultDataset

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    def _batch_num(p: Path) -> int:
        m = re.fullmatch(r"batch_(\d+)\.npy", p.name)
        return int(m.group(1)) if m else -1

    batch_files = sorted(
        [p for p in input_dir.glob("batch_*.npy") if re.fullmatch(r"batch_\d+\.npy", p.name)],
        key=_batch_num,
    )

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

    # Stream through numpy batches one at a time, verifying the corresponding
    # Lance rows by position.  Memory usage is bounded by the largest batch file.
    lance_offset = 0
    with tqdm(total=actual_count, desc="Verifying", unit="records") as pbar:
        for batch_path in batch_files:
            batch_data = np.load(batch_path)
            batch_size = len(batch_data)

            # Small per-batch lookup — freed when the loop advances to the next batch
            id_to_tensor: dict[str, np.ndarray] = {str(row["id"]): row["tensor"] for row in batch_data}

            for j in range(batch_size):
                lance_idx = lance_offset + j
                obj_id = dataset.get_object_id(lance_idx)
                if obj_id not in id_to_tensor:
                    raise RuntimeError(
                        f"Row {lance_idx}: object_id {obj_id!r} found in Lance but not in {batch_path.name}."
                    )
                print(dataset[lance_idx])
                lance_tensor = np.asarray(dataset[lance_idx]["data"])
                numpy_tensor = np.asarray(id_to_tensor[obj_id])
                tensors_match_bitwise = (
                    lance_tensor.shape == numpy_tensor.shape
                    and lance_tensor.dtype == numpy_tensor.dtype
                    and np.ascontiguousarray(lance_tensor).tobytes()
                    == np.ascontiguousarray(numpy_tensor).tobytes()
                )
                if not tensors_match_bitwise:
                    raise RuntimeError(
                        f"Row {lance_idx} (id={obj_id!r}): tensor mismatch.\n"
                        f"  Lance : {lance_tensor}\n"
                        f"  NumPy : {numpy_tensor}"
                    )
                pbar.update(1)

            lance_offset += batch_size

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
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the verification step after conversion (not recommended).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        convert(input_dir, output_dir)
        if not args.skip_verify:
            verify(input_dir, output_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
