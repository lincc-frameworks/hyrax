import logging
import warnings

from colorama import Back, Fore, Style

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Train(Verb):
    """Train verb"""

    cli_name = "train"
    add_parser_kwargs = {}

    # Dataset groups that the Train verb knows about.
    # REQUIRED_DATA_GROUPS must be present in the dataset dict returned by setup_dataset.
    # OPTIONAL_DATA_GROUPS are used when present but do not cause an error if absent.
    REQUIRED_DATA_GROUPS = ("train",)
    OPTIONAL_DATA_GROUPS = ("validate", "test")

    @staticmethod
    def setup_parser(parser):
        """We don't need any parser setup for CLI opts"""
        pass

    def run_cli(self, args=None):
        """CLI stub for Train verb"""
        logger.info("train run from CLI.")

        self.run()

    def run(self):
        """
        Run the training process for the configured model and data loader.
        Returns the trained model.

        """

        import mlflow

        from hyrax.config_utils import create_results_dir, log_runtime_config
        from hyrax.gpu_monitor import GpuMonitor
        from hyrax.pytorch_ignite import (
            create_trainer,
            create_validator,
            dist_data_loader,
            setup_dataset,
            setup_model,
        )
        from hyrax.tensorboardx_logger import close_tensorboard_logger, init_tensorboard_logger

        config = self.config

        # Create a results directory
        results_dir = create_results_dir(config, "train")
        log_runtime_config(config, results_dir)

        # Create a tensorboardX logger
        init_tensorboard_logger(log_dir=results_dir)

        # Instantiate the model and dataset
        dataset = setup_dataset(
            config,
            splits=Train.REQUIRED_DATA_GROUPS + Train.OPTIONAL_DATA_GROUPS,
        )
        model = setup_model(config, dataset["train"])
        logger.info(
            f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Training model:{Style.RESET_ALL} "
            f"{model.__class__.__name__}"
        )
        logger.info(f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Training dataset(s):{Style.RESET_ALL}\n{dataset}")

        # We know that `dataset` will always be returned as a dictionary with at least
        # a `train` and `infer` key. There may be a `validate` key as well.
        #
        # There are three ways splits can be defined:
        #
        # 1) Separate dataset groups: the user defined distinct "train" and
        #    "validate" groups in their data_request (possibly pointing to
        #    different data_locations).  Each DataProvider is loaded
        #    independently and we pass split=False to dist_data_loader.
        #
        # 2) split_fraction on shared data: the user defined "train" and
        #    "validate" groups pointing to the *same* data_location with
        #    split_fraction values.  setup_dataset has already computed
        #    non-overlapping split_indices on each DataProvider, so
        #    dist_data_loader with split=False will automatically apply a
        #    SubsetSequentialSampler.  This path is handled identically to (1).
        #
        # 3) Legacy percentage-based splits: only a "train" group exists and
        #    no split_fraction is set.  We fall back to the old behaviour of
        #    calling dist_data_loader with split=["train", "validate"] which
        #    uses config["data_set"] train_size / validate_size.

        # Collect split names in two ways:
        # - all_splits: all split names that this verb knows about
        #   (required + optional), used for legacy percentage-based
        #   splitting where only a "train" group may be defined.
        # - dataset_splits: those desired splits that are actually present
        #   in the dataset dict returned by setup_dataset, used by the
        #   multi-provider path where each split is an explicit group.
        all_splits = list(Train.REQUIRED_DATA_GROUPS) + list(Train.OPTIONAL_DATA_GROUPS)
        dataset_splits = [s for s in all_splits if s in dataset]

        # Check whether split_fraction was used (path 2 above).
        # This is true when the required split's DataProvider has split_indices assigned.
        # Path 1 (separate groups without split_fraction) will be handled in the else block.
        has_split_groups = isinstance(dataset, dict) and any(
            hasattr(dataset.get(s), "split_indices") and dataset[s].split_indices is not None
            for s in Train.REQUIRED_DATA_GROUPS
        )

        data_loaders: dict[str, tuple] = {}

        if has_split_groups:
            # Path 2: split_fraction was used — each DataProvider has split_indices.
            # Create a dataloader per group with split_indices already applied.
            # NOTE: Paths 1 and 3 will be completely deprecated in a future release,
            # and this will be the only path for training.
            for split_name in dataset_splits:
                data_loaders[split_name] = dist_data_loader(dataset[split_name], config, False)
        elif len(dataset) > 1:
            # Path 1: separate dataset groups defined in data_request without split_fraction.
            # Each group is an independent DataProvider pointing to different data_locations.
            # Create a dataloader per group.
            for split_name in dataset_splits:
                data_loaders[split_name] = dist_data_loader(dataset[split_name], config, split_name)
        else:
            # Path 3 (legacy): only "train" exists — use percentage-based
            # splitting from config["data_set"].
            warnings.warn(
                "Defining dataset splits via config['data_set'] "
                "(train_size / validate_size / test_size) is deprecated. "
                "Please define separate dataset groups with 'split_fraction' "
                "in the [data_request] configuration instead. "
                "See https://hyrax.readthedocs.io for migration guidance.",
                DeprecationWarning,
                stacklevel=1,
            )
            raw = dist_data_loader(dataset["train"], config, all_splits)
            # dist_data_loader returns a bare (DataLoader, indices) tuple
            # when given a single split name, or a dict when given multiple.
            if isinstance(raw, dict):
                for split_name in all_splits:
                    if split_name in raw:
                        data_loaders[split_name] = raw[split_name]
            else:
                # Single split — raw is already the (DataLoader, indices) tuple.
                data_loaders[all_splits[0]] = raw

        train_data_loader, _ = data_loaders["train"]
        validation_data_loader, _ = data_loaders.get("validate", (None, None))

        # Create trainer, a pytorch-ignite `Engine` object
        trainer = create_trainer(model, config, results_dir)

        # Create a validator if a validation data loader is available
        if validation_data_loader is not None:
            create_validator(model, config, validation_data_loader, trainer)

        monitor = GpuMonitor()

        # Go up to the parent of the results dir so all mlflow results show up in the same directory.
        mlflow_dir = (results_dir.parent / "mlflow").resolve()
        mlflow.set_tracking_uri("file://" + str(mlflow_dir))

        # Get experiment_name and cast to string (it's a tomlkit.string by default)
        experiment_name = str(config["train"]["experiment_name"])

        # This will create the experiment if it doesn't exist
        mlflow.set_experiment(experiment_name)

        # If run_name is not `false` in the config, use it as the MLFlow run name in
        # this experiment. Otherwise use the name of the results directory
        run_name = str(config["train"]["run_name"]) if config["train"]["run_name"] else results_dir.name

        with mlflow.start_run(log_system_metrics=True, run_name=run_name):
            Train._log_params(config, results_dir)

            # Run the training process
            trainer.run(train_data_loader, max_epochs=config["train"]["epochs"])

        # Save the trained model
        model.save(results_dir / config["train"]["weights_filename"])
        monitor.stop()

        logger.info("Finished Training")
        close_tensorboard_logger()

        return model

    @staticmethod
    def _log_params(config, results_dir):
        """Log the various parameters to mlflow from the config file.

        Parameters
        ----------
        config : dict
            The main configuration dictionary

        results_dir: str
            The full path to the results sub-directory
        """
        import mlflow

        # Log full path to results subdirectory
        mlflow.log_param("Results Directory", results_dir)

        # Log all model params
        mlflow.log_params(config["model"])

        # Log some training and data loader params
        mlflow.log_param("epochs", config["train"]["epochs"])
        mlflow.log_param("batch_size", config["data_loader"]["batch_size"])

        # Log the criterion and optimizer params
        criterion_name = config["criterion"]["name"]
        mlflow.log_param("criterion", criterion_name)
        if criterion_name in config:
            mlflow.log_params(config[criterion_name])

        optimizer_name = config["optimizer"]["name"]
        mlflow.log_param("optimizer", optimizer_name)
        if optimizer_name in config:
            mlflow.log_params(config[optimizer_name])
