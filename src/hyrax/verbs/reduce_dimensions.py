import gc
import logging
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import numpy as np

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class ReduceDimensions(Verb):
    """Verb to reduce the dimensionality of a dataset"""

    # Use an attribute-friendly name so `hyrax.reduce_dimensions` resolves.
    cli_name = "reduce_dimensions"
    add_parser_kwargs = {}
    description = "Reduce the dimensionality of a dataset using provided or default reduction algorithm."

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """Setup parser for reduce-dimensions verb"""
        parser.add_argument(
            "-a",
            "--algorithm",
            type=str,
            required=False,
            help="Dimensionality reduction algorithm to use (default: umap).",
        )
        parser.add_argument(
            "-i",
            "--input-dir",
            type=str,
            required=False,
            help="Directory containing the dataset to reduce dimensions for.",
        )
        parser.add_argument(
            "-m",
            "--model-path",
            type=str,
            required=False,
            help="Path to a pre-existing reducer model.",
        )

    def run_cli(self, args: Namespace | None = None):
        """CLI stub for ReduceDimensions verb"""
        logger.info("`reduce-dimensions` run from CLI.")

        if args is None:
            raise RuntimeError("Run CLI called with no arguments.")

        return self.run(algorithm=args.algorithm, input_dir=args.input_dir, model_path=args.model_path)

    def run(
        self,
        algorithm: str | None = None,
        input_dir: Union[Path, str] | None = None,
        model_path: Union[Path, str] | None = None,
    ):
        """
        Run dimensionality reduction on a dataset

        This method loads the latent space representations from an inference run and applies
        the selected dimensionality reduction algorithm.

        Algorithms that support reusable fitted models may either:
        - fit a new model using a sampled subset of the data, or
        - load an existing model if a model path is provided.

        Algorithms without a separate fitting stage do not support model loading and
        directly transform the input data.

        The full dataset is then transformed into the target lower-dimensional space,
        and the resulting embeddings are saved.

        Parameters
        ----------
        algorithm : str, Optional
            The dimensionality reduction algorithm to use.
            If not specified, the method will look in the config for a default algorithm.

        input_dir : str or Path, Optional
            Directory containing the dataset to reduce dimensions for.

        model_path : str or Path, Optional
            Path to a pre-existing reducer model.

        Returns
        -------
        None
            The method does not return anything but saves the algorithm reducer representations to disk.
        """
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            return self._run(algorithm, input_dir, model_path)

    def _run(
        self, algorithm: str | None, input_dir: Union[Path, str] | None, model_path: Union[Path, str] | None
    ):
        """See run()"""
        from hyrax.config_utils import create_results_dir
        from hyrax.datasets.result_factories import create_results_writer, load_results_dataset
        from hyrax.verbs.reduction_algorithms.algorithm_registry import fetch_reducer_class

        # Get reducer class
        algorithm_name = algorithm or self.config["reduce"]["algorithm"]
        reducer_cls = fetch_reducer_class(algorithm_name)

        results_dir = create_results_dir(self.config, f"{algorithm_name}")
        logger.info(f"Saving reduction results using {algorithm_name} to {results_dir}")
        reduction_results = create_results_writer(results_dir)

        algo_reducer = reducer_cls(self.config, reduction_results)

        inference_results = load_results_dataset(self.config, results_dir=input_dir, verb="infer")
        total_length = len(inference_results)

        # Prepare data sample for either fitting a new model or validating a pre-trained model loaded.
        config_sample_size = self.config["reduce"][algorithm_name].get("fit_sample_size", None)
        sample_size = int(np.min([config_sample_size if config_sample_size else np.inf, total_length]))
        rng = np.random.default_rng()
        sample_indexes = rng.choice(np.arange(total_length), size=sample_size, replace=False)
        data_sample = np.asarray(inference_results[sample_indexes]).reshape((sample_size, -1))

        # Load model if path provided, otherwise fit new model
        # Getting the model of current algorithm specified.
        if model_path is None:
            model_path = self.config["reduce"][algorithm_name].get("model_path", None)

        if model_path:
            logger.info(f"Loading pre-existing reducer model from {model_path}")
            algo_reducer.load_model(data_sample.shape[1], model_path)
        else:
            logger.info("No model_path specified. A new model will be fitted.")
            algo_reducer.fit(data_sample)

            if self.config["reduce"].get("save_fit_model", False):
                logger.info(f"Saving fitted {algorithm_name} reducer to result directory")
                algo_reducer.save_model(results_dir)

        del data_sample
        gc.collect()

        # Transform dataset
        batch_size = self.config["reduce"]["batch_size"]
        num_batches = int(np.ceil(total_length / batch_size))

        all_indexes = np.arange(0, total_length)
        all_ids = np.array(inference_results.ids())

        args = (
            (
                all_ids[batch_indexes],
                inference_results[batch_indexes].reshape(len(batch_indexes), -1),
            )
            for batch_indexes in np.array_split(all_indexes, num_batches)
        )
        algo_reducer.transform(args, num_batches)

        logger.info(f"Finished transforming all data with {algorithm_name}")

        return load_results_dataset(self.config, results_dir)
