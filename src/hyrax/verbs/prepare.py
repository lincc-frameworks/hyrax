import logging

from hyrax.pytorch_ignite import setup_dataset

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Prepare(Verb):
    """Prepare verb"""

    cli_name = "prepare"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser):
        """We don't need any parser setup for CLI opts"""
        pass

    def run_cli(self, args=None):
        """CLI stub for Prepare verb"""
        logger.error("Prepare is not supported from the command line.")

    def run(self, config):
        """Prepare the dataset for a given model and data loader.

        Parameters
        ----------
        config : dict
            The parsed config file as a nested
            dict
        """

        data_set = setup_dataset(config)

        logger.info("Finished Prepare")
        return data_set
