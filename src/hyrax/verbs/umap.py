import logging
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

from .reduce_dimensions import ReduceDimensions
from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Umap(Verb):
    """Umap latent space points into 2d"""

    cli_name = "umap"
    add_parser_kwargs = {}
    description = "Transforms the entire dataset into a lower-dimensional space by fitting a UMAP model."

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """Stub of parser setup"""
        parser.add_argument(
            "-i",
            "--input-dir",
            type=str,
            required=False,
            help="Directory containing inference results to umap.",
        )

        parser.add_argument(
            "-m",
            "--model-path",
            type=str,
            required=False,
            help="Path to a pre-existing UMAP model.",
        )

    # Should there be a version of this on the base class which uses a dict on the Verb
    # superclass to build the call to run based on what the subclass verb defined in setup_parser
    def run_cli(self, args: Namespace | None = None):
        """Stub CLI implementation"""
        logger.info("umap run from cli")
        if args is None:
            raise RuntimeError("Run CLI called with no arguments.")

        # This is where we map from CLI parsed args to a
        # self.run (args) call.
        return self.run(input_dir=args.input_dir, model_path=args.model_path)

    def run(self, input_dir: Union[Path, str] | None = None, model_path: Union[Path, str] | None = None):
        """
        Deprecated wrapper for reduce_dimensions running the UMAP algorithm.

        This wrapper delegates execution to ``reduce_dimensions`` with
        ``algorithm='umap'`` so that ``umap`` verb remains available for backward compatibility.
        But users are encouraged to switch to using
        ``reduce_dimensions``.

        Parameters
        ----------
        input_dir : str or Path, Optional
            The directory containing the inference results.

        model_path : str or Path, Optional
            The path to a pre-existing UMAP model.

        Returns
        -------
        None
            The method does not return anything but saves the UMAP representations to disk.
        """
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            return self._run(input_dir, model_path)

    def _run(self, input_dir: Union[Path, str] | None = None, model_path: Union[Path, str] | None = None):
        """See run()"""
        logger.warning("The `umap` verb is deprecated. Use `reduce_dimensions(algorithm='umap')` instead.")
        return ReduceDimensions(self.config).run(algorithm="umap", input_dir=input_dir, model_path=model_path)
