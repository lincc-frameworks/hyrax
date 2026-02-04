import logging

from colorama import Back, Fore, Style

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Infer(Verb):
    """Inference verb"""

    cli_name = "infer"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser):
        """We don't need any parser setup for CLI opts"""
        pass

    def run_cli(self, args=None):
        """CLI stub for Infer verb"""
        logger.info("infer run from CLI")

        self.run()

    def run(self):
        """Run inference on a model using a dataset

        Parameters
        ----------
        config : dict
            The parsed config file as a nested dict
        """

        from hyrax.config_utils import (
            create_results_dir,
            log_runtime_config,
        )
        from hyrax.data_sets.inference_dataset import InferenceDataSet
        from hyrax.models.model_utils import load_model_weights
        from hyrax.pytorch_ignite import (
            create_evaluator,
            create_save_batch_callback,
            dist_data_loader,
            setup_dataset,
            setup_model,
        )
        from hyrax.tensorboardx_logger import close_tensorboard_logger, init_tensorboard_logger

        config = self.config
        context = {}

        # Create a results directory and dump our config there
        results_dir = create_results_dir(config, "infer")

        # Create a tensorboardX logger
        init_tensorboard_logger(log_dir=results_dir)

        dataset = setup_dataset(config)
        model = setup_model(config, dataset["infer"])
        logger.info(
            f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Inference model:{Style.RESET_ALL} "
            f"{model.__class__.__name__}"
        )
        logger.info(
            f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}Inference dataset(s):{Style.RESET_ALL}\n{dataset}"
        )

        # Inference doesnt work at all with the dataloader doing additional shuffling:
        if config["data_loader"]["shuffle"]:
            msg = "Data loader shuffling not supported in inference mode. "
            msg += "Setting config['data_loader']['shuffle'] = False"
            logger.warning(msg)
            config["data_loader"]["shuffle"] = False

        # If `dataset` is a dict containing the key "infer", we'll pull that out.
        # The only time it wouldn't be is if the dataset is an iterable dataset.
        if isinstance(dataset, dict) and "infer" in dataset:
            dataset = dataset["infer"]
            if dataset.is_map():
                logger.debug(f"Inference dataset has length: {len(dataset)}")  # type: ignore[arg-type]

        data_loader, _ = dist_data_loader(dataset, config, False)

        load_model_weights(config, model, "infer")
        log_runtime_config(config, results_dir)
        context["results_dir"] = results_dir

        # Log Results directory
        logger.info(f"Saving inference results at: {results_dir}")

        model.save(results_dir / "inference_weights.pth")

        # Create the save batch callback
        save_batch_callback = create_save_batch_callback(dataset, results_dir)

        # Run inference
        evaluator = create_evaluator(model, save_batch_callback, config)
        evaluator.run(data_loader)

        # Write out a dictionary to map IDs->Batch
        save_batch_callback.data_writer.write_index()  # type: ignore[attr-defined]

        # Write out our tensorboard stuff
        close_tensorboard_logger()

        # Log completion
        logger.info("Inference Complete.")

        return InferenceDataSet(config, results_dir)
