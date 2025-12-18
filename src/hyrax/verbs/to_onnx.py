import logging

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class ToOnnx(Verb):
    """Export the model to ONNX format"""

    cli_name = "to_onnx"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser):
        """Setup parser for ONNX export verb"""
        parser.add_argument(
            "--input-model-directory",
            type=str,
            required=False,
            help="Directory containing the trained model to export.",
        )

    def run_cli(self, args=None):
        """Run the ONNX export verb from the CLI"""
        logger.info("Exporting model to ONNX format.")
        self.run(args.input_model_directory)

    def run(self, input_model_directory: str = None):
        """Export the model to ONNX format and save it to the specified path."""
        import shutil
        from pathlib import Path

        from hyrax.config_utils import (
            ConfigManager,
            create_results_dir,
            find_most_recent_results_dir,
            log_runtime_config,
        )
        from hyrax.model_exporters import export_to_onnx
        from hyrax.pytorch_ignite import dist_data_loader, setup_dataset, setup_model

        config = self.config

        # Resolve the input directory in this order; 1) input_model_directory arg,
        # 2) config value, 3) most recent train results
        if input_model_directory:
            input_directory = Path(input_model_directory)
            if not input_directory.exists():
                logger.error(f"Input model directory {input_directory} does not exist.")
                return
        elif config["onnx"]["input_model_directory"]:
            input_directory = Path(config["onnx"]["input_model_directory"])
            if not input_directory.exists():
                logger.error(f"Input model directory in the config file {input_directory} does not exist.")
                return
        else:
            input_directory = find_most_recent_results_dir(config, "train")
            if not input_directory:
                logger.error("No previous training results directory found for ONNX export.")
                return

        output_dir = create_results_dir(config, "onnx")
        log_runtime_config(config, output_dir)

        # grab the config file from the input directory, and render it.
        config_file = input_directory / "runtime_config.toml"
        config_manager = ConfigManager(runtime_config_filepath=config_file)
        config_from_training = config_manager.config

        # copy the to_tensor.py file from the input directory to the output directory
        to_tensor_src = input_directory / "to_tensor.py"
        to_tensor_dst = output_dir / "to_tensor.py"
        if to_tensor_src.exists():
            shutil.copy(to_tensor_src, to_tensor_dst)

        # Use the config file to locate and assemble the trained weight file path
        weights_file_path = input_directory / config_from_training["train"]["weights_filename"]

        if not weights_file_path.exists():
            raise RuntimeError(f"Could not find trained model weights: {weights_file_path}")

        # Use the config in the model directory to load the dataset(s) and create
        # The data loader instance to provide a data sample to the ONNX exporter.
        dataset = setup_dataset(config_from_training)
        model = setup_model(config_from_training, dataset["infer"])
        # Load the trained weights and send the model to the CPU for ONNX export.
        model.load(weights_file_path)
        model.train(False)

        # Create an instance of the dataloader so that we can request a sample batch.
        infer_data_loader, _ = dist_data_loader(dataset["infer"], config_from_training, False)

        # Generate the `context` dictionary that will be provided to the ONNX exporter.
        context = {
            "results_dir": output_dir,
            "ml_framework": "pytorch",
        }

        # Get a sample of input data.
        batch_sample = next(iter(infer_data_loader))
        batch_sample = model.to_tensor(batch_sample)

        export_to_onnx(model, batch_sample, config, context)
