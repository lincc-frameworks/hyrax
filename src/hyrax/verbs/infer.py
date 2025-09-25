import logging
from pathlib import Path
from typing import Optional, Union

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
        config : ConfigDict
            The parsed config file as a nested dict
        """
        import inspect

        import numpy as np
        from tensorboardX import SummaryWriter
        from torch import Tensor

        from hyrax.config_utils import (
            create_results_dir,
            log_runtime_config,
        )
        from hyrax.data_sets.inference_dataset import InferenceDataSet, InferenceDataSetWriter
        from hyrax.pytorch_ignite import (
            create_evaluator,
            dist_data_loader,
            setup_dataset,
            setup_model,
        )

        config = self.config
        context = {}

        # Create a results directory and dump our config there
        results_dir = create_results_dir(config, "infer")

        # Create a tensorboardX logger
        tensorboardx_logger = SummaryWriter(log_dir=results_dir)

        dataset = setup_dataset(config, tensorboardx_logger)
        model = setup_model(config, dataset)
        if dataset.is_map():
            logger.debug(f"data set has length {len(dataset)}")  # type: ignore[arg-type]

        # Inference doesnt work at all with the dataloader doing additional shuffling:
        if config["data_loader"]["shuffle"]:
            msg = "Data loader shuffling not supported in inference mode. "
            msg += "Setting config['data_loader']['shuffle'] = False"
            logger.warning(msg)
            config["data_loader"]["shuffle"] = False

        data_loader, data_loader_indexes = dist_data_loader(dataset, config, split=config["infer"]["split"])

        Infer.load_model_weights(config, model)
        log_runtime_config(config, results_dir)
        context["results_dir"] = results_dir

        # Log Results directory
        logger.info(f"Saving inference results at: {results_dir}")

        with open(results_dir / "to_tensor.py", "w") as f:
            try:
                f.write(inspect.getsource(model.to_tensor))
            except (OSError, TypeError) as e:
                logger.warning(f"Could not retrieve source for model.to_tensor: {e}")
                f.write("# Source code for model.to_tensor could not be retrieved.\n")

        data_writer = InferenceDataSetWriter(dataset, results_dir)

        # These are values the _save_batch callback needs to run
        write_index = 0
        object_ids = np.array(list(dataset.ids()))[data_loader_indexes]  # type: ignore[attr-defined]

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
                batch_is_list and isinstance(batch.get(0), Tensor)
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

        # Run inference
        evaluator = create_evaluator(model, _save_batch, config)
        evaluator.run(data_loader)

        # Write out a dictionary to map IDs->Batch
        data_writer.write_index()

        # Write out our tensorboard stuff
        tensorboardx_logger.close()

        # Log completion
        logger.info("Inference Complete.")

        return InferenceDataSet(config, results_dir)

    @staticmethod
    def load_model_weights(config, model):
        """Loads the model weights from a file. Raises RuntimeError if this is not possible due to
        config, missing or malformed file

        Parameters
        ----------
        config : ConfigDict
            Full runtime configuration
        model : nn.Module
            The model class to load weights into

        """
        from hyrax.config_utils import find_most_recent_results_dir

        weights_file: Optional[Union[str, Path]] = (
            config["infer"]["model_weights_file"] if config["infer"]["model_weights_file"] else None
        )

        if weights_file is None:
            recent_results_path = find_most_recent_results_dir(config, "train")
            if recent_results_path is None:
                raise RuntimeError("Must define model_weights_file in the [infer] section of hyrax config.")

            weights_file = recent_results_path / config["train"]["weights_filename"]

        # Ensure weights file is a path object.
        weights_file_path = Path(weights_file)

        if not weights_file_path.exists():
            raise RuntimeError(f"Model Weights file {weights_file_path} does not exist")

        try:
            model.load(weights_file_path)
            config["infer"]["model_weights_file"] = str(weights_file_path)
        except Exception as err:
            msg = f"Model weights file {weights_file_path} did not load properly. Are you sure you are "
            msg += "predicting using the correct model"
            raise RuntimeError(msg) from err
