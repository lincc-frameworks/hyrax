import logging
from argparse import ArgumentParser, Namespace

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Search(Verb):
    """Stub of similarity search"""

    cli_name = "search"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """Stub of parser setup"""
        parser.add_argument("-i", "--image-file", type=str, help="Path to image file", required=True)

    # If both of these move to the verb superclass then a new verb is basically
    #
    # If you want no args, just make the class, define run(self)
    # If you want args
    #     1) write setup_parser (which sets up for ArgumentParser and name/type info for cli run)
    #     2) write run(self, <your args>) to do what you want
    #

    # Should there be a version of this on the base class which uses a dict on the Verb
    # superclass to build the call to run based on what the subclass verb defined in setup_parser
    def run_cli(self, args: Namespace | None = None):
        """Stub CLI implementation"""
        logger.info("Search run from cli")
        if args is None:
            raise RuntimeError("Run CLI called with no arguments.")
        # This is where we map from CLI parsed args to a
        # self.run (args) call.
        return self.run(image_file=args.image_file)

    def run(self, image_file: str):
        """Search for... todo

        Parameters
        ----------
        image_file : str
            _description_
        """
        logger.info(f"Got Image {image_file}")
