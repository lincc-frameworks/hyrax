import logging

from colorama import Back, Fore, Style

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Train(Verb):
    """Train verb"""

    cli_name = "train"
    add_parser_kwargs = {}

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

        from hyrax.config_utils import create_results_dir, log_mlflow_params, log_runtime_config
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
        dataset = setup_dataset(config)
        model = setup_model(config, dataset["train"])
        logger.info(
            f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Training model:{Style.RESET_ALL} "
            f"{model.__class__.__name__}"
        )
        logger.info(f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Training dataset(s):{Style.RESET_ALL}\n{dataset}")

        # We know that `dataset` will always be returned as a dictionary with at least
        # a `train` and `infer` key. There may be a `validate` key as well.
        # The only instance in which a dataset would not be a dictionary is if
        # the user has requested an iterable dataset. But we don't want to support that
        # for training right now.
        if isinstance(dataset, dict) and "validate" in dataset:
            train_data_loader, _ = dist_data_loader(dataset["train"], config, False)
            validation_data_loader, _ = dist_data_loader(dataset["validate"], config, False)

        # if `validate` isn't in the dataset dict, then we assume the user wants to
        # use percentage-based splits on the `train` dataset. Or the user has an
        # iterable dataset - but we don't support training with iterable datasets.
        else:
            data_loaders = dist_data_loader(dataset["train"], config, ["train", "validate"])
            train_data_loader, _ = data_loaders["train"]
            validation_data_loader, _ = data_loaders.get("validate", (None, None))

        # Create trainer, a pytorch-ignite `Engine` object
        trainer = create_trainer(model, config, results_dir)

        # Create a validator if a validation data loader is available
        if validation_data_loader is not None:
            create_validator(model, config, results_dir, validation_data_loader, trainer)

        monitor = GpuMonitor()

        mlflow.set_tracking_uri("file://" + str(results_dir / "mlflow"))

        # Get experiment_name and cast to string (it's a tomlkit.string by default)
        experiment_name = str(config["train"]["experiment_name"])

        # This will create the experiment if it doesn't exist
        mlflow.set_experiment(experiment_name)

        # If run_name is not `false` in the config, use it as the MLFlow run name in
        # this experiment. Otherwise use the name of the results directory
        run_name = str(config["train"]["run_name"]) if config["train"]["run_name"] else results_dir.name

        with mlflow.start_run(log_system_metrics=True, run_name=run_name):
            log_mlflow_params(config, results_dir, "train")

            # Run the training process
            trainer.run(train_data_loader, max_epochs=config["train"]["epochs"])

        # Save the trained model
        model.save(results_dir / config["train"]["weights_filename"])
        monitor.stop()

        logger.info("Finished Training")
        close_tensorboard_logger()

        return model
