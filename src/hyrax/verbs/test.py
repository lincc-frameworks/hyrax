import logging
from pathlib import Path

from colorama import Back, Fore, Style

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Test(Verb):
    """Test verb - evaluates a trained model on test data"""

    cli_name = "test"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser):
        """We don't need any parser setup for CLI opts"""
        pass

    def run_cli(self, args=None):
        """CLI stub for Test verb"""
        logger.info("test run from CLI.")

        self.run()

    def run(self):
        """
        Run the test process for the configured model on test data.
        This evaluates a trained model, saves outputs, and returns metrics.

        Note: The configuration dictionary will be updated with the full path to the
        model weights file that is loaded into the model (config["test"]["model_weights_file"]).

        Returns
        -------
        InferenceDataSet
            Dataset containing test results that can be used for further analysis
        """

        import mlflow
        from tensorboardX import SummaryWriter

        from hyrax.config_utils import (
            create_results_dir,
            log_runtime_config,
        )
        from hyrax.data_sets.result_factories import load_results_dataset
        from hyrax.models.model_utils import load_model_weights
        from hyrax.pytorch_ignite import (
            create_evaluator,
            create_save_batch_callback,
            create_tester,
            dist_data_loader,
            setup_dataset,
            setup_model,
        )

        config = self.config

        # Create a results directory
        results_dir = create_results_dir(config, "test")

        # Create a tensorboardX logger
        tensorboardx_logger = SummaryWriter(log_dir=results_dir)

        # Instantiate the model and dataset
        dataset = setup_dataset(config)

        # Verify that test dataset exists
        if not isinstance(dataset, dict) or "test" not in dataset:
            raise RuntimeError("No test dataset available. Please configure a test dataset or split.")

        model = setup_model(config, dataset["test"])

        # Load model weights
        load_model_weights(config, model, "test")

        # Log runtime config after loading weights so the actual weights path is captured
        log_runtime_config(config, results_dir)

        logger.info(
            f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Testing model:{Style.RESET_ALL} "
            f"{model.__class__.__name__}"
        )
        logger.info(
            f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Test dataset:{Style.RESET_ALL}\n{dataset['test']}"
        )

        # Disable shuffling for test (like inference)
        if config["data_loader"]["shuffle"]:
            msg = "Data loader shuffling not supported in test mode. "
            msg += "Setting config['data_loader']['shuffle'] = False"
            logger.warning(msg)
            config["data_loader"]["shuffle"] = False

        # Determine which dataset to use for testing
        test_data_loader, _ = dist_data_loader(dataset["test"], config, False)

        # Save the loaded model weights to the test results directory
        model.save(results_dir / "test_weights.pth")

        # Create the save batch callback
        save_batch_callback = create_save_batch_callback(dataset["test"], results_dir)

        results_root_dir = Path(config["general"]["results_dir"]).expanduser().resolve()
        mlflow.set_tracking_uri("file://" + str(results_root_dir / "mlflow"))

        # Get experiment_name from train config
        experiment_name = str(config["train"]["experiment_name"])

        # This will create the experiment if it doesn't exist
        mlflow.set_experiment(experiment_name)

        # Use run_name if provided, otherwise use results directory name
        run_name = config["test"].get("run_name") if config["test"].get("run_name") else results_dir.name

        with mlflow.start_run(log_system_metrics=True, run_name=run_name):
            Test._log_params(config, results_dir)

            # Create two engines: one for metrics, one for saving outputs
            # First, run evaluator to save model outputs
            evaluator_engine = create_evaluator(model, save_batch_callback, config)
            evaluator_engine.run(test_data_loader)

            # Then, run tester to compute metrics
            test_engine = create_tester(model, config)
            test_engine.run(test_data_loader)

        # Write out a dictionary to map IDs->Batch
        save_batch_callback.data_writer.commit()  # type: ignore[attr-defined]

        logger.info("Finished Testing")
        tensorboardx_logger.close()

        # Return the ResultDataset for further analysis
        return load_results_dataset(config, results_dir)

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

        # Log batch size
        mlflow.log_param("batch_size", config["data_loader"]["batch_size"])

        # Log the criterion params
        criterion_name = config["criterion"]["name"]
        mlflow.log_param("criterion", criterion_name)
        if criterion_name in config:
            mlflow.log_params(config[criterion_name])
