import logging
from pathlib import Path

import ignite.distributed as idist
import torch
from colorama import Back, Fore, Style

from hyrax.trace import trace_verb_data

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Train(Verb):
    """Train verb"""

    cli_name = "train"
    add_parser_kwargs = {}
    description = "Train a model using provided data."

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

    @trace_verb_data
    def run(self):
        """
        Run the training process for the configured model and data loader.
        Returns the trained model.

        """

        import mlflow

        from hyrax.config_utils import create_results_dir, log_runtime_config
        from hyrax.gpu_monitor import GpuMonitor
        from hyrax.pytorch_ignite import (
            Events,
            attach_best_checkpoint,
            create_trainer,
            create_validator,
            dist_data_loader,
            setup_dataset,
            setup_model,
        )
        from hyrax.splitting_utils import create_splits
        from hyrax.tensorboardx_logger import close_tensorboard_logger, init_tensorboard_logger

        config = self.config

        # Validate that the user hasn't set both `resume` and `model_weights_file`.
        # These are mutually exclusive: `resume` restores a full training checkpoint
        # (model weights, optimizer state, epoch counter), while `model_weights_file`
        # loads only model parameters and starts training fresh.
        if config["train"]["resume"] and config["train"]["model_weights_file"]:
            raise ValueError(
                "Cannot set both `resume` and `model_weights_file` in the [train] config. "
                "Use `resume` to continue from a full checkpoint (restores optimizer state and epoch), "
                "or use `model_weights_file` to start fresh training from pre-trained weights."
            )

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
        create_splits(config, dataset, results_dir=results_dir, persist=True)
        model = setup_model(config, dataset["train"])

        def training(rank):
            logger.info(
                f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Training model:{Style.RESET_ALL} "
                f"{model.__class__.__name__}"
            )
            logger.info(
                f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Training dataset(s):{Style.RESET_ALL}\n{dataset}"
            )

            # If a pre-trained weights file is specified, load it before creating the trainer.
            # This must happen before create_trainer() wraps the model with idist.auto_model
            # (the distributed wrapper) to avoid parameter key name mismatches.
            if config["train"]["model_weights_file"]:
                from hyrax.models.model_utils import load_model_weights

                load_model_weights(config, model, "train")
                logger.info(
                    f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Loading pre-trained weights:"
                    f"{Style.RESET_ALL} {config['train']['model_weights_file']}"
                )
                logger.info(
                    f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Fine-tuning mode:{Style.RESET_ALL} "
                    "Training will start from epoch 1 with a fresh optimizer."
                )

            train_shuffle = config["train"]["shuffle"]

            dataset_splits = [
                s for s in Train.REQUIRED_DATA_GROUPS + Train.OPTIONAL_DATA_GROUPS if s in dataset
            ]

            data_loaders: dict[str, tuple] = {}
            for split_name in dataset_splits:
                data_loaders[split_name] = dist_data_loader(
                    dataset[split_name],
                    config,
                    shuffle=split_name == "train" and train_shuffle,
                )

            train_data_loader = data_loaders["train"]
            validation_data_loader = data_loaders.get("validate")

            # Create trainer, a pytorch-ignite `Engine` object
            trainer = create_trainer(model, config, results_dir)

            # Dispatch on_epoch_start to all DataProviders at the start of each epoch.
            @trainer.on(Events.EPOCH_STARTED)
            def dispatch_epoch_start(engine):
                for provider in dataset.values():
                    provider.on_epoch_start("train")

            # Create a validator if a validation data loader is available
            if validation_data_loader is not None:
                validator = create_validator(model, config, validation_data_loader, trainer)
                attach_best_checkpoint(validator, model, trainer, results_dir)
            else:
                attach_best_checkpoint(trainer, model, trainer, results_dir)

            monitor = GpuMonitor()

            # Go up to the parent of the results dir so all mlflow results show up in the same directory.
            results_root_dir = Path(config["general"]["results_dir"]).expanduser().resolve()
            (results_root_dir / "mlflow").mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri("sqlite:///" + str(results_root_dir / "mlflow" / "mlflow.db"))

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

        nproc_per_node = (
            torch.cuda.device_count() if torch.cuda.is_available()
            else torch.multiprocessing.cpu_count()
        )
        with idist.Parallel(backend="nccl", nproc_per_node=nproc_per_node) as parallel:
            parallel.run(training)

        # training(0)
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
