import logging
from pathlib import Path
from typing import Union

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

        Returns
        -------
        InferenceDataSet
            Dataset containing test results that can be used for further analysis
        """

        import mlflow
        import numpy as np
        from tensorboardX import SummaryWriter
        from torch import Tensor

        from hyrax.config_utils import create_results_dir, log_runtime_config
        from hyrax.data_sets.inference_dataset import InferenceDataSet, InferenceDataSetWriter
        from hyrax.pytorch_ignite import (
            create_tester,
            dist_data_loader,
            setup_dataset,
            setup_model,
        )

        config = self.config

        # Create a results directory
        results_dir = create_results_dir(config, "test")
        log_runtime_config(config, results_dir)

        # Create a tensorboardX logger
        tensorboardx_logger = SummaryWriter(log_dir=results_dir)

        # Instantiate the model and dataset
        dataset = setup_dataset(config, tensorboardx_logger)
        
        # Verify that test dataset exists
        if not isinstance(dataset, dict) or "test" not in dataset:
            raise RuntimeError("No test dataset available. Please configure a test dataset or split.")
        
        model = setup_model(config, dataset["test"])
        logger.info(
            f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Testing model:{Style.RESET_ALL} "
            f"{model.__class__.__name__}"
        )
        logger.info(f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Test dataset(s):{Style.RESET_ALL}\n{dataset}")

        # Determine which dataset to use for testing
        test_data_loader, data_loader_indexes = dist_data_loader(dataset["test"], config, False)

        # Load model weights
        Test.load_model_weights(config, model)

        # Save the loaded model weights to the test results directory
        model.save(results_dir / "test_weights.pth")

        # Initialize writer for saving test outputs
        data_writer = InferenceDataSetWriter(dataset["test"], results_dir)

        # These are values the _save_batch callback needs to run
        write_index = 0
        object_ids = np.array(list(dataset["test"].ids()))[data_loader_indexes]  # type: ignore[attr-defined]

        def _save_batch(batch: Union[Tensor, list, tuple, dict], batch_results: Tensor):
            """Receive and write results tensors to results_dir immediately
            This function writes a single numpy binary file for each object.
            """
            nonlocal write_index
            nonlocal object_ids
            nonlocal data_writer

            batch_len = len(batch_results)
            batch_results = batch_results.detach().to("cpu")

            batch_is_list = isinstance(batch, (tuple, list))
            # Batch lacks ids if it is a Tensor, or a list/tuple of tensors
            batch_lacks_ids = isinstance(batch, Tensor) or (
                batch_is_list and isinstance(batch[0] if batch_is_list else None, Tensor)
            )

            # Batch has IDs if it is dict of tensors with the needed key
            batch_has_ids = isinstance(batch, dict) and "object_id" in batch
            if batch_lacks_ids:
                # This fallback is brittle to any re-ordering of data that occurs during data loading
                batch_object_ids = [
                    object_ids[id] for id in range(write_index, write_index + len(batch_results))
                ]
            elif batch_has_ids:
                if isinstance(batch["object_id"], list):
                    batch_object_ids = batch["object_id"]
                else:
                    batch_object_ids = batch["object_id"].tolist()
            elif isinstance(batch, dict):
                msg = "Dataset dictionary should be returning object_ids to avoid ordering errors. "
                msg += "Modify the __getitem__ or __iter__ function of your dataset to include 'object_id' "
                msg += "with unique values per data member in the dictionary it returns."
                raise RuntimeError(msg)
            else:
                msg = f"Could not determine object IDs from batch. Batch has type {type(batch)}"
                raise RuntimeError(msg)

            # Save results from this batch in a numpy file as a structured array
            data_writer.write_batch(np.array(batch_object_ids), [t.numpy() for t in batch_results])
            write_index += batch_len

        results_root_dir = Path(config["general"]["results_dir"]).expanduser().resolve()
        mlflow.set_tracking_uri("file://" + str(results_root_dir / "mlflow"))

        # Get experiment_name and cast to string
        experiment_name = str(
            config.get("test", {}).get("experiment_name", config["train"]["experiment_name"])
        )

        # This will create the experiment if it doesn't exist
        mlflow.set_experiment(experiment_name)

        # Use run_name if provided, otherwise use results directory name
        run_name_config = config.get("test", {}).get("run_name", False)
        run_name = str(run_name_config) if run_name_config else results_dir.name

        with mlflow.start_run(log_system_metrics=True, run_name=run_name):
            Test._log_params(config, results_dir)

            # Create test evaluator with save function and run test
            test_evaluator = create_tester(model, config, results_dir, tensorboardx_logger, _save_batch)
            test_evaluator.run(test_data_loader)

        # Write out a dictionary to map IDs->Batch
        data_writer.write_index()

        logger.info("Finished Testing")
        tensorboardx_logger.close()

        # Return the InferenceDataSet for further analysis
        return InferenceDataSet(config, results_dir)

    @staticmethod
    def load_model_weights(config, model):
        """Loads the model weights from a file. Raises RuntimeError if this is not possible due to
        config, missing or malformed file

        Parameters
        ----------
        config : dict
            Full runtime configuration
        model : nn.Module
            The model class to load weights into

        """
        from typing import Union

        from hyrax.config_utils import find_most_recent_results_dir

        # The "test" key will always exist in config
        weights_file: Union[str, Path] | None = (
            config["test"]["model_weights_file"] if config["test"]["model_weights_file"] else None
        )

        if weights_file is None:
            recent_results_path = find_most_recent_results_dir(config, "train")
            if recent_results_path is None:
                raise RuntimeError("Must define model_weights_file in the [test] section of hyrax config.")

            weights_file = recent_results_path / config["train"]["weights_filename"]

        # Ensure weights file is a path object.
        weights_file_path = Path(weights_file)

        if not weights_file_path.exists():
            raise RuntimeError(f"Model Weights file {weights_file_path} does not exist")

        try:
            model.load(weights_file_path)
            # Update config to track which weights file was actually used
            config["test"]["model_weights_file"] = str(weights_file_path)
        except Exception as err:
            msg = f"Model weights file {weights_file_path} did not load properly. Are you sure you are "
            msg += "testing with the correct model?"
            raise RuntimeError(msg) from err

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
