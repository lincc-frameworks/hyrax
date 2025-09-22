import logging
from pathlib import Path

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
        import inspect

        import mlflow
        from tensorboardX import SummaryWriter

        from hyrax.config_utils import create_results_dir, log_runtime_config
        from hyrax.gpu_monitor import GpuMonitor
        from hyrax.model_exporters import export_to_onnx
        from hyrax.pytorch_ignite import (
            create_trainer,
            create_validator,
            dist_data_loader,
            setup_dataset,
            setup_model,
        )

        config = self.config

        # Create a results directory
        results_dir = create_results_dir(config, "train")
        log_runtime_config(config, results_dir)

        # Create a tensorboardX logger
        tensorboardx_logger = SummaryWriter(log_dir=results_dir)

        # Instantiate the model and dataset
        dataset = setup_dataset(config, tensorboardx_logger)
        model = setup_model(config, dataset)

        # Create a data loader for the training set (and validation split if configured)
        data_loaders = dist_data_loader(dataset, config, ["train", "validate"])
        train_data_loader, _ = data_loaders["train"]
        validation_data_loader, _ = data_loaders.get("validate", (None, None))

        # Create trainer, a pytorch-ignite `Engine` object
        trainer = create_trainer(model, config, results_dir, tensorboardx_logger)

        # Create a validator if a validation data loader is available
        if validation_data_loader is not None:
            create_validator(model, config, results_dir, tensorboardx_logger, validation_data_loader, trainer)

        monitor = GpuMonitor(tensorboard_logger=tensorboardx_logger)

        results_root_dir = Path(config["general"]["results_dir"]).expanduser().resolve()
        mlflow.set_tracking_uri("file://" + str(results_root_dir / "mlflow"))

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
        with open(results_dir / "to_tensor.py", "w") as f:
            try:
                f.write(inspect.getsource(model.to_tensor))
            except (OSError, TypeError) as e:
                logger.warning(f"Could not retrieve source for model.to_tensor: {e}")
                f.write("# Source code for model.to_tensor could not be retrieved.\n")
        monitor.stop()

        logger.info("Finished Training")
        tensorboardx_logger.close()

        context = {
            "ml_framework": "pytorch",
            "results_dir": results_dir,
        }

        # Get a sample of input data. If the data is labeled, only return the input data.
        batch_sample = next(iter(train_data_loader))
        if isinstance(batch_sample, dict):
            batch_sample = model.to_tensor(batch_sample)
        sample = batch_sample[0] if isinstance(batch_sample, (list, tuple)) else batch_sample

        export_to_onnx(model, sample, config, context)

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
