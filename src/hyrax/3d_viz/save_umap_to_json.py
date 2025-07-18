import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table, join
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_umap_json(
    results_dir,
    output_json="umap_data.json",
    fits_table_path=None,
    columns_to_include=None,
    id_column="object_id",
    join_type="left",
    keep_first_match_only=False,
):
    """
    Saves UMAP 3D embeddings and object IDs into a JSON file for use in Three.js
    Also provides the option of including metadata from any Astropy readable table

    Parameters
    ----------
    results_dir : str or Path
        Directory containing the UMAP embedding batch files
    output_json : str, optional
        Output JSON filename, default is "umap_data.json"
    fits_table_path : str or Path, optional
        Path to a table in a file format that is supported by Astropy FITS table
        or pickle file (FITS table or Pandas dataframe) for cross-matching with
        additional properties
    columns_to_include : list, optional
        List of column names from the FITS table to include in the JSON output.
        If None and fits_table_path is provided, all columns will be included.
    id_column : str, optional
        Name of the column in the FITS table to use for ID matching, default is "object_id"
    join_type : str, optional
        Type of join to perform with the FITS table. Options are:
        - 'left': Include all UMAP points, with FITS data when available (default)
        - 'inner': Include only UMAP points that have matching FITS data
        - 'outer': Include all UMAP points and all FITS data, with nulls when no match
        - 'right': Include all FITS rows, with UMAP data when available
    keep_first_match_only : bool, optional
        If True, when multiple rows in the FITS table match a single object_id,
        only the first match will be kept. Default is False.
    """

    # Validate join_type parameter
    valid_join_types = ["left", "right", "inner", "outer"]
    if join_type not in valid_join_types:
        logger.error(f"join_type must be one of {valid_join_types}, got '{join_type}'")

    results_dir = Path(results_dir)

    # Find batch files matching 'batch_<number>.npy'
    batch_files = sorted([f for f in results_dir.glob("batch_*.npy") if re.match(r"batch_\d+\.npy$", f.name)])

    if not batch_files:
        raise FileNotFoundError(
            f"No valid batch files found in {results_dir}. Ensure the directory\
                                contains files matching batch_<number>.npy"
        )

    embeddings_list = []
    object_ids_list = []

    for batch_file in tqdm(batch_files, desc="Loading batch files"):
        data = np.load(batch_file)
        embeddings_list.append(data["tensor"])
        object_ids_list.append(data["id"])

    # Concatenate all embeddings and object IDs
    embeddings = np.concatenate(embeddings_list, axis=0)
    object_ids = np.concatenate(object_ids_list, axis=0)

    # Create an Astropy table with the embeddings and object IDs
    umap_table = Table()
    umap_table["id"] = object_ids
    umap_table["x"] = embeddings[:, 0]
    umap_table["y"] = embeddings[:, 1]
    umap_table["z"] = embeddings[:, 2]

    # Load FITS table if provided
    if fits_table_path is not None:
        fits_table_path = Path(fits_table_path)
        if not fits_table_path.exists():
            logger.error(f"Table file not found: {fits_table_path}")

        logger.info(f"Loading table file: {fits_table_path}")

        # Check if the file is a pickle file by examining its extension
        file_ext = fits_table_path.suffix.lower()

        if file_ext in [".pkl", ".pickle"]:
            import pickle

            # Load the pickle file
            logger.info(f"Detected pickle file format: {file_ext}")
            with open(fits_table_path, "rb") as f:
                # If it's a pandas DataFrame, convert to Astropy Table
                pickle_data = pickle.load(f)
                if isinstance(pickle_data, pd.DataFrame):
                    fits_table = Table.from_pandas(pickle_data)
                    logger.info("Converted pickled pandas DataFrame to Astropy Table")
                elif isinstance(pickle_data, Table):
                    fits_table = pickle_data
                    logger.info("Loaded pickled Astropy Table")
                else:
                    raise TypeError(
                        f"Pickled data must be either pandas DataFrame\
                                        or Astropy Table, got {type(pickle_data)}"
                    )
        else:
            # Try to load as a standard Astropy table format
            fits_table = Table.read(fits_table_path)

        # Verify the table has the specified ID column for cross-matching
        if id_column not in fits_table.colnames:
            raise ValueError(f"Table must contain a '{id_column}' column for cross-matching")

        # Check for duplicate IDs in the FITS table
        id_counts = Counter(fits_table[id_column])
        duplicate_ids = {id_val: count for id_val, count in id_counts.items() if count > 1}

        if duplicate_ids and keep_first_match_only:
            total_duplicates = sum(count - 1 for count in duplicate_ids.values())
            num_duplicated_ids = len(duplicate_ids)
            total_rows = len(fits_table)

            logger.info(
                f"Found {num_duplicated_ids} IDs with multiple rows ({total_duplicates} duplicate\
                    rows out of {total_rows} total rows)"
            )
            logger.info("Using keep_first_match_only=True, only the first occurrence of each ID will be kept")

            # Create a new table with only the first occurrence of each ID
            seen_ids = set()
            rows_to_keep = []

            for row in fits_table:
                current_id = row[id_column]
                if current_id not in seen_ids:
                    rows_to_keep.append(row)
                    seen_ids.add(current_id)

            # Create a new table with only the selected rows
            fits_table = Table(rows=rows_to_keep, names=fits_table.colnames)
            logger.info(f"Reduced FITS table from {total_rows} to {len(fits_table)} rows")
        elif duplicate_ids:
            total_duplicates = sum(count - 1 for count in duplicate_ids.values())
            num_duplicated_ids = len(duplicate_ids)

            logger.info(
                f"Warning: Found {num_duplicated_ids} IDs with multiple rows ({total_duplicates}\
                            duplicate rows)"
            )
            logger.info(
                "This will result in duplicate points in the output JSON. Use\
                            keep_first_match_only=True to keep only first matches."
            )

            # Example of duplicates for the user to understand
            if num_duplicated_ids > 0:
                example_id = list(duplicate_ids.keys())[0]
                example_count = duplicate_ids[example_id]
                logger.info(f"Example: ID {example_id} appears {example_count} times")

        # If columns_to_include is None, use all columns except the ID column
        if columns_to_include is None:
            columns_to_include = [
                col for col in fits_table.colnames if (col != id_column and len(fits_table[col].shape) <= 1)
            ]
            logger.info(f"Including all table columns except multidimensional columns: {columns_to_include}")
        else:
            # Verify requested columns exist in the table
            missing_columns = [col for col in columns_to_include if col not in fits_table.colnames]
            if missing_columns:
                raise ValueError(f"Requested columns not found in table: {missing_columns}")

        # Select only the columns we need from the FITS table
        columns_to_keep = [id_column] + columns_to_include
        fits_table = fits_table[columns_to_keep]

        # Rename the ID column in the FITS table to match the umap_table
        fits_table.rename_column(id_column, "id")

        # Check the data type of the FITS table ID column
        fits_id_dtype = fits_table["id"].dtype

        # Convert UMAP table ID column to match the FITS table type
        logger.info(f"Converting UMAP ID column to match FITS table ID type: {fits_id_dtype}")
        try:
            umap_table["id"] = umap_table["id"].astype(fits_id_dtype)
            logger.info("ID column conversion successful")
        except Exception as e:
            logger.info(f"Error converting ID column: {e}")
            logger.info("Falling back to string conversion for both tables")
            umap_table["id"] = umap_table["id"].astype(str)
            fits_table["id"] = fits_table["id"].astype(str)

        # Perform the join using Astropy's built-in functionality
        logger.info(f"Cross-matching using '{id_column}' column with join_type='{join_type}'...")
        joined_table = join(umap_table, fits_table, keys="id", join_type=join_type)

        # Report match statistics based on join type
        if join_type in ["left", "outer"]:
            total_umap_points = len(umap_table)
            missing_fits_data = sum(
                1
                for row in joined_table
                if any(
                    np.ma.is_masked(row[col]) for col in columns_to_include if col in joined_table.colnames
                )
            )
            matched_points = total_umap_points - missing_fits_data
            logger.info(
                f"UMAP points with matching table data: {matched_points} out of\
                         {total_umap_points} ({matched_points / total_umap_points * 100:.1f}%)"
            )

        if join_type in ["right", "outer"]:
            total_fits_rows = len(fits_table)
            missing_umap_data = sum(
                1
                for row in joined_table
                if any(np.ma.is_masked(row[col]) for col in ["x", "y", "z"] if col in joined_table.colnames)
            )
            matched_fits = total_fits_rows - missing_umap_data
            logger.info(
                f"Table rows with matching UMAP data: {matched_fits} out of\
                            {total_fits_rows} ({matched_fits / total_fits_rows * 100:.1f}%)"
            )

        if join_type == "inner":
            logger.info(f"Inner join results: {len(joined_table)} matching points")

        # Check if the join resulted in more rows than original UMAP points (meaning duplicates)
        if len(joined_table) > len(umap_table) and not keep_first_match_only:
            logger.info(
                f"Join resulted in {len(joined_table)} rows, which is more than the\
                         {len(umap_table)} UMAP points"
            )
            logger.info(
                "This indicates some UMAP points matched multiple table rows.\
                            The output will contain duplicate points."
            )

        # Convert to pandas DataFrame for easier handling of masked values
        df = joined_table.to_pandas()

        # Replace masked values with None for proper JSON serialization
        df = df.replace({pd.NA: None})

        # Convert numpy types to Python native types for JSON serialization
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Only convert to float for actual float columns, preserve ints
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = df[col].astype("int64")  # or leave as is
                else:
                    df[col] = df[col].astype("float64")
    else:
        # If no FITS table provided, just convert UMAP data to pandas
        df = umap_table.to_pandas()

    # Convert to JSON format
    logger.info("Creating JSON data structure...")

    # First convert DataFrame to records style dict
    records = df.to_dict(orient="records")

    # Process each record to handle any remaining special types
    json_data = {"points": []}
    for record in tqdm(records, desc="Formatting JSON"):
        # Process any remaining numpy arrays or other special types
        for k, v in record.items():
            if isinstance(v, np.ndarray):
                record[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                record[k] = float(v)
            elif isinstance(v, bytes):
                try:
                    # Try to decode bytes to string
                    record[k] = v.decode("utf-8")
                except UnicodeDecodeError:
                    # If it can't be decoded, convert to base64 string
                    import base64

                    record[k] = base64.b64encode(v).decode("ascii")
            elif pd.isna(v):
                record[k] = None
        json_data["points"].append(record)

    # Save to file
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"UMAP data saved to {output_json}")

    # Report overall statistics
    logger.info(f"Total points in JSON: {len(json_data['points'])}")
    if fits_table_path is not None:
        columns = ["x", "y", "z", "id"] + [c for c in columns_to_include if c in df.columns]
        logger.info(f"Each point has properties: {', '.join(columns)}")


if __name__ == "__main__":
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create argument parser
    parser = argparse.ArgumentParser(description="Save UMAP 3D embeddings to JSON")

    # Required arguments
    parser.add_argument("results_dir", type=str, help="Directory containing the UMAP embedding batch files")

    # Optional arguments
    parser.add_argument(
        "--output_json",
        "-o",
        type=str,
        default="umap_data.json",
        help="Output JSON filename (default: umap_data.json)",
    )
    parser.add_argument(
        "--fits_table_path",
        "-f",
        type=str,
        default=None,
        help="Path to a table format accepted by Astropy Tables or pickle file for\
                            cross-matching and incorproation of additional metadata",
    )
    parser.add_argument(
        "--columns_to_include",
        "-c",
        type=str,
        nargs="+",
        default=None,
        help="List of column names from the fits-table to include",
    )
    parser.add_argument(
        "--id_column",
        "-i",
        type=str,
        default="object_id",
        help="Name of the ID column for matching to the fits-table (default: object_id)",
    )
    parser.add_argument(
        "--join_type",
        "-j",
        type=str,
        default="left",
        choices=["left", "right", "inner", "outer"],
        help="Type of join to perform (default: left)",
    )
    parser.add_argument(
        "--keep_first_match_only",
        "-k",
        action="store_true",
        help="Keep only the first match when multiple rows match an ID in the fits table",
    )

    args = parser.parse_args()

    # Run the function with the provided arguments
    save_umap_json(
        results_dir=args.results_dir,
        output_json=args.output_json,
        fits_table_path=args.fits_table_path,
        columns_to_include=args.columns_to_include,
        id_column=args.id_column,
        join_type=args.join_type,
        keep_first_match_only=args.keep_first_match_only,
    )
