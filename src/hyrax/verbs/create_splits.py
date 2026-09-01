import logging

from hyrax.trace import trace_verb_data

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class CreateSplits(Verb):
    """Create and persist reproducible dataset splits."""

    cli_name = "create_splits"
    add_parser_kwargs = {}
    description = "Compute and persist dataset splits for reproducible training workflows."

    REQUIRED_DATA_GROUPS = ()
    OPTIONAL_DATA_GROUPS = ()

    @staticmethod
    def setup_parser(parser):
        """No additional CLI options needed."""

    def run_cli(self, args=None):
        """CLI stub for CreateSplits verb."""
        logger.info("create_splits run from CLI")
        self.run()

    @trace_verb_data
    def run(self):
        """Compute dataset splits and write them to a results directory.

        Reads the ``[split]`` and ``[balance]`` config tables to determine how to
        partition each data group, then persists ``.npz`` index files and a
        ``split_config.toml`` under a timestamped ``*-splits-*`` results directory.
        Subsequent verbs (``train``, ``infer``, ``test``) can point at this directory
        to reuse the same split without recomputing it.

        Returns
        -------
        dict[str, DataProvider]
            The populated dataset providers, keyed by group name.
        """
        from hyrax.config_utils import create_results_dir, log_runtime_config
        from hyrax.pytorch_ignite import setup_dataset
        from hyrax.splitting_utils import create_splits

        config = self.config

        results_dir = create_results_dir(config, "splits")
        datasets = setup_dataset(config)
        create_splits(config, datasets, results_dir=results_dir, persist=True)
        log_runtime_config(config, results_dir)

        logger.info(f"Split files written to: {results_dir}")
        return datasets
