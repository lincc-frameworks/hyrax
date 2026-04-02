import logging

from hyrax.pytorch_ignite import setup_dataset

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Prepare(Verb):
    """Prepare Verb, Prepares a dataset and returns it"""

    cli_name = "prepare"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser):
        """We don't need any parser setup for CLI opts"""
        pass

    def run_cli(self, args=None):
        """CLI stub for Prepare verb"""
        logger.info("Prepare run from CLI")

        retval = self.run()
        print(retval)

    def run(self):
        """Prepare the dataset for a given model and data loader using the verb's configuration.

        Uses ``self.config`` to construct and return the prepared dataset.
        """
        data_set = setup_dataset(self.config)
        logger.info("Finished Prepare")
        return data_set
