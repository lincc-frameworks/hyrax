import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class DatabaseConnection(Verb):
    """Verb to insert inference results into a vector database index for fast
    similarity search."""

    cli_name = "database_connection"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser: ArgumentParser):
        """Stub of parser setup"""

        parser.add_argument(
            "-d",
            "--database-dir",
            type=str,
            required=False,
            help="Directory of existing vector database.",
        )

    def run_cli(self, args: Optional[Namespace] = None):
        """Stub CLI implementation"""
        logger.error("Database connection is not supported from the command line.")

    def run(self, database_dir: Optional[Union[Path, str]] = None):
        """Create a connection to the vector database for interactive queries.

        Parameters
        ----------
        database_dir : str or Path, Optional
            The directory containing the database that will be connected to.
            If None, attempt to connect to the most recently created `...-vector-db-...`
            directory. If specified, it can point to either an empty directory
            or a directory containing an existing vector database. If the latter, the
            database will be updated with the new vectors.
        """
        from hyrax.config_utils import find_most_recent_results_dir
        from hyrax.vector_dbs.vector_db_factory import vector_db_factory

        config = self.config

        # Attempt to find the directory containing the vector database. Check for
        # the database_dir argument first, then check the config file for
        # vector_db.vector_db_dir, and finally check for the most recently
        # created vector-db directory.
        vector_db_dir = None
        if database_dir is not None:
            vector_db_dir = database_dir
        elif config["vector_db"]["vector_db_dir"]:
            vector_db_dir = config["vector_db"]["vector_db_dir"]
        else:
            vector_db_dir = find_most_recent_results_dir(config, "vector-db")

        vector_db_path = Path(vector_db_dir).resolve()
        if not vector_db_path.is_dir():
            raise RuntimeError(
                f"Database directory {str(vector_db_path)} does not exist. \
                    Have you run `hyrax.save_to_database(output_dir={vector_db_path})`?"
            )

        # Get the flavor of database (i.e. Chroma, Qdrant, etc) from the config
        # file saved in `vector_db_path`. This ensures that we will use the correct
        # database class when creating the connection.
        db_type = self._get_database_type_from_config(vector_db_path)
        config["vector_db"]["name"] = db_type

        # Create an instance of the vector database class for the connection
        self.vector_db = vector_db_factory(config, context={"results_dir": vector_db_path})
        if self.vector_db is None:
            raise RuntimeError(f"Unable to conenct to the {db_type} database in directory {vector_db_path}")

        return self.vector_db

    def _get_database_type_from_config(self, database_dir: Path):
        """Internal function that will read a config file from a directory and
        return the name of the vector database from it. i.e. "chromadb", "qdrant".

        Parameters
        ----------
        database_dir : Path
            The directory containing the vector database and the config file that
            be used as reference.

        Returns
        -------
        str
            The config value for ["vector_db"]["name"] in the reference config.
        """
        from hyrax.config_utils import ConfigManager

        config_file = database_dir / "runtime_config.toml"
        reference_config = ConfigManager.read_runtime_config(config_filepath=config_file)
        return reference_config["vector_db"]["name"]
