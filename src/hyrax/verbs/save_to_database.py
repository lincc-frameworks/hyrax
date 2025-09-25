import logging
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tensorboardX import SummaryWriter

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class SaveToDatabase(Verb):
    """Verb to insert inference results into a vector database index for fast
    similarity search."""

    cli_name = "save_to_database"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """Stub of parser setup"""
        parser.add_argument(
            "-i",
            "--input-dir",
            type=str,
            required=False,
            help="Directory containing inference results to index.",
        )

        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            required=False,
            help="Directory of existing vector database, if adding more vectors.",
        )

    def run_cli(self, args: Optional[Namespace] = None):
        """Stub CLI implementation"""
        logger.info("Creating vector db index from cli")
        if args is None:
            raise RuntimeError("Run CLI called with no arguments.")

        return self.run(input_dir=args.input_dir, output_dir=args.output_dir)

    def run(
        self, input_dir: Optional[Union[Path, str]] = None, output_dir: Optional[Union[Path, str]] = None
    ):
        """Insert inference results into vector database.

        Parameters
        ----------
        input_dir : str or Path, Optional
            The directory containing the inference results.

        output_dir : str or Path, Optional
            The directory where the vector database is stored. If None, a new directory
            will be created. If specified, it can point to either an empty directory
            or a directory containing an existing vector database. If the latter, the
            database will be updated with the new vectors.
        """
        from copy import deepcopy

        from tqdm import tqdm

        from hyrax.config_utils import (
            create_results_dir,
            find_most_recent_results_dir,
            log_runtime_config,
        )
        from hyrax.data_sets.inference_dataset import InferenceDataSet
        from hyrax.vector_dbs.vector_db_factory import vector_db_factory

        config = deepcopy(self.config)

        # Attempt to find the directory containing inference results. Check for
        # the --input-dir argument first, then check the config file for
        # vector_db.infer_results_dir, and finally check for the most recent
        # results directory.
        infer_results_dir = None
        if input_dir is not None:
            infer_results_dir = input_dir
        elif config["vector_db"]["infer_results_dir"]:
            infer_results_dir = config["vector_db"]["infer_results_dir"]
        else:
            infer_results_dir = find_most_recent_results_dir(config, "infer")

        if infer_results_dir is None:
            raise RuntimeError("Must define infer_results_dir in the [vector_db] section of hyrax config.")

        inference_results_path = Path(infer_results_dir).resolve()
        if not inference_results_path.is_dir():
            raise RuntimeError(f"Input directory {inference_results_path} does not exist.")

        # Create an instance of the InferenceDataSet
        inference_data_set = InferenceDataSet(config, inference_results_path)

        # Get the vector db output directory by using the --output-dir parameter or
        # config value or creating a new directory, in that order.
        vector_db_dir = Path()
        if output_dir is not None:
            vector_db_dir = output_dir
        elif config["vector_db"]["vector_db_dir"]:
            vector_db_dir = config["vector_db"]["vector_db_dir"]
        else:
            vector_db_dir = create_results_dir(config, "vector-db")

        vector_db_path = Path(vector_db_dir).resolve()
        if not vector_db_path.is_dir():
            raise RuntimeError(f"Database directory {str(vector_db_path)} does not exist.")

        logger.info(f"Saving vector database at {vector_db_dir}")

        # Create an instance of the vector database to insert into
        vector_db = vector_db_factory(config, context={"results_dir": str(vector_db_path)})
        if vector_db:
            vector_db.create()
        else:
            raise RuntimeError(
                "No vector database configured. "
                "Please specify a supported vector db in the ['vector_db']['name'] "
                "section of the hyrax config."
            )

        # Log the config with updated values for the input and output directories.
        config["vector_db"]["infer_results_dir"] = str(inference_results_path)
        config["vector_db"]["vector_db_dir"] = str(vector_db_path)
        log_runtime_config(config, vector_db_path)

        # Create a tensorboardX logger for metrics
        tensorboardx_logger = SummaryWriter(log_dir=vector_db_path)

        # Use the batch_index to get the list of batches.
        batches = np.unique(inference_data_set.batch_index["batch_num"])

        logger.debug(f"Number of inference result batches to index: {len(batches)}.")

        total_insertion_time = 0.0
        batch_count = 0

        for batch in tqdm(batches):
            # Get all the indices where inference_data_set.batch_index['batch_num'] == batch
            index_mask = inference_data_set.batch_index["batch_num"] == batch

            # Get the ids of the data in this batch file
            ids = inference_data_set.batch_index["id"][index_mask]

            # Retrieve the vectors from the batch file using the ids. We use the
            # ids here so that we only have to open one file to get the vectors.
            inference_data = inference_data_set._load_from_batch_file(batch, ids)

            # Flatten the vectors and turn them into a list of np.arrays.
            vectors = list(inference_data["tensor"].reshape(len(inference_data["tensor"]), -1))

            # Time the vector database insertion
            start_time = time.time()
            vector_db.insert(ids=list(inference_data["id"]), vectors=vectors)
            insertion_time = time.time() - start_time

            # Log insertion metrics to Tensorboard
            batch_count += 1
            total_insertion_time += insertion_time
            vectors_inserted = len(vectors)

            tensorboardx_logger.add_scalar("vector_db/batch_insertion_time", insertion_time, batch_count)
            tensorboardx_logger.add_scalar("vector_db/vectors_per_batch", vectors_inserted, batch_count)
            rate = vectors_inserted / insertion_time if insertion_time > 0 else 0
            tensorboardx_logger.add_scalar("vector_db/insertion_rate_vectors_per_second", rate, batch_count)

            logger.debug(
                f"Batch {batch}: Inserted {vectors_inserted} vectors in {insertion_time:.3f}s "
                f"({vectors_inserted / insertion_time:.1f} vectors/sec)"
            )

        # Log total insertion metrics
        tensorboardx_logger.add_scalar("vector_db/total_insertion_time", total_insertion_time, 1)
        tensorboardx_logger.add_scalar("vector_db/total_batches", batch_count, 1)
        avg_time = total_insertion_time / batch_count if batch_count > 0 else 0
        tensorboardx_logger.add_scalar("vector_db/average_batch_insertion_time", avg_time, 1)

        # Close the tensorboard logger
        tensorboardx_logger.close()

        logger.info(
            f"Vector database insertion complete. Total time: {total_insertion_time:.3f}s "
            f"for {batch_count} batches"
        )
