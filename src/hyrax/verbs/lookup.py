import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import numpy as np

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Lookup(Verb):
    """Look up an inference result using the ID of a data member"""

    cli_name = "lookup"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """Set up our arguments by configuring a subparser

        Parameters
        ----------
        parser : ArgumentParser
            The sub-parser to configure
        """
        parser.add_argument("-i", "--id", type=str, required=True, help="ID of image")
        parser.add_argument(
            "-r", "--results-dir", type=str, required=False, help="Directory containing inference results."
        )

    def run_cli(self, args: Namespace | None = None):
        """Entrypoint to Lookup from the CLI.

        Parameters
        ----------
        args : Optional[Namespace], optional
            The parsed command line arguments

        """
        logger.info("Lookup run from cli")
        if args is None:
            raise RuntimeError("Run CLI called with no arguments.")
        # This is where we map from CLI parsed args to a
        # self.run (args) call.
        vector = self.run(id=args.id, results_dir=args.results_dir)
        if vector is None:
            logger.info("No inference result found")
        else:
            logger.info("Inference result found")
            print(vector)

    def run(self, id: str, results_dir: Union[Path, str] | None = None) -> np.ndarray | None:
        """Lookup the latent-space representation of a particular ID

        Requires the relevant dataset to be configured, and for inference to have been run.

        Parameters
        ----------
        id : str
            The ID of the input data to look up the inference result

        results_dir : str, Optional
            The directory containing the inference results.

        Returns
        -------
        Optional[np.ndarray]
            The output tensor of the model for the given input.
        """
        from hyrax.config_utils import find_most_recent_results_dir
        from hyrax.data_sets.inference_dataset import InferenceDataSet

        if results_dir is None:
            if self.config["results"]["inference_dir"]:
                results_dir = self.config["results"]["inference_dir"]
            else:
                results_dir = find_most_recent_results_dir(self.config, verb="infer")
                msg = f"Using most recent results dir {results_dir} for lookup."
                msg += "Use the [results] inference_dir config to set a directory or pass it to this verb."
                logger.info(msg)

        if results_dir is None:
            msg = "Could not find a results directory. Run infer or use "
            msg += "[results] inference_dir config to specify a directory"
            logger.error(msg)
            return None

        if isinstance(results_dir, str):
            results_dir = Path(results_dir)

        inference_dataset = InferenceDataSet(self.config, results_dir=results_dir)

        all_ids = np.array(list(inference_dataset.ids()))
        lookup_index = np.argwhere(all_ids == id)

        if len(lookup_index) == 1:
            return np.array(inference_dataset[lookup_index[0]].numpy())
        elif len(lookup_index) > 1:
            raise RuntimeError(f"Inference result directory {results_dir} has duplicate ID numbers")

        return None
