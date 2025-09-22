import logging

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Model(Verb):
    """Resolves the model class that is defined in the config file.
    This will return a reference to the model class."""

    cli_name = "model"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser):
        """Not implemented"""
        pass

    def run_cli(self):
        """Not implemented"""
        logger.error("Running model from the cli is unimplemented")

    def run(self):
        """Fetch and return the model _class_. Does not create an instance of
        the model class.
        """
        from hyrax.models.model_registry import fetch_model_class

        config = self.config

        model_cls = fetch_model_class(config)

        return model_cls
