import logging

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class Engine(Verb):
    """This verb drives inference with an ONNX model in production."""

    cli_name = "engine"
    add_parser_kwargs = {}

    @staticmethod
    def setup_parser(parser):
        """Setup parser for engine verb"""
        parser.add_argument(
            "--model-directory",
            type=str,
            required=False,
            help="Directory containing the ONNX model.",
        )

    def run_cli(self, args=None):
        """CLI stub for Engine verb"""
        logger.info("`engine` run from CLI.")

        self.run(model_directory=args.model_directory if args else None)

    def run(self, model_directory: str = None):
        """
        [x] Read in the user config
        [x] Prepare all the datasets requested
        [x] Implement a simple strategy for reading in batches of data samples
        [ ] Process the samples with any custom collate functions as well as a default collate function
        [x] Pass the collated batch to the appropriate to_tensor function
        [ ] Send that output to the ONNX-ified model
        [x] Persist the results of inference.
        """
        from pathlib import Path

        import onnxruntime

        from hyrax.config_utils import (
            create_results_dir,
            find_most_recent_results_dir,
        )
        from hyrax.data_sets.inference_dataset import InferenceDatasetWriter
        from hyrax.plugin_utils import load_to_tensor
        from hyrax.pytorch_ignite import setup_dataset

        config = self.config

        # ~ Find the directory that contains the ONNX model, to_tensor.py, etc.
        if model_directory:
            input_directory = Path(model_directory)
            if not input_directory.exists():
                logger.error(f"Model directory {input_directory} does not exist.")
                return
        elif config["engine"]["model_directory"]:
            input_directory = Path(config["engine"]["model_directory"])
            if not input_directory.exists():
                logger.error(f"Model directory in the config file {input_directory} does not exist.")
                return
        else:
            input_directory = find_most_recent_results_dir(config, "onnx")
            if not input_directory:
                logger.error("No previous training results directory found for ONNX export.")
                return

        # ~ Here we load the appropriate to_tensor function from onnx output.
        to_tensor_fn = load_to_tensor(input_directory)

        # ~ Load the ONNX model from the input directory.
        onnx_file_name = input_directory / "model.onnx"
        ort_session = onnxruntime.InferenceSession(onnx_file_name)

        # ~ For now we use `setup_dataset` to get our datasets back. Later we can
        # optimize this, because we know that we'll only need the `infer` part
        # of the model_inputs dictionary. And we can assume that we'll be working
        # with map-style datasets. But for now, this gets us going.
        dataset = setup_dataset(config)

        # ~ In the `train` and `infer` verbs, we use `dist_data_loader` to create
        # our data loaders. But here in `engine`, we can assume that we can simply
        # find the length of our dataset and then iterate over it in batches.
        infer_dataset = dataset["infer"]
        batch_size = config["data_loader"]["batch_size"]

        # ~ Initialize the InferenceDatasetWriter to persist results of inference
        # Note that the inference_dataset.py module takes a dependency on
        # torch.utils.data.Dataset, but InferenceDatasetWrite only uses Dataset
        # as a type hint. So we may need to separate InferenceDataset and IDWriter
        # to remove that dependency.
        result_dir = create_results_dir(config, "engine")
        self.results_writer = InferenceDatasetWriter(infer_dataset, result_dir)

        # Work through the dataset in steps of `batch_size`
        for start_idx in range(0, len(infer_dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(infer_dataset))
            batch = [infer_dataset[i] for i in range(start_idx, end_idx)]

            # ~ Here is where we can process the batch with any custom
            # collate functions as well as a default collate function.
            # This is left as a TODO until we do the work in DataProvider to maintain
            # a map of collate functions for each dataset.
            collated_batch = batch  # default_collate_function(batch)

            # Since the DataProvider also maintains the model_inputs definition
            # It is the logical place to put a default collate function.

            # ~ Pass the collated batch to the to_tensor function
            prepared_batch = to_tensor_fn(collated_batch)

            # Then we would send that output to the ONNX-ified model.
            ort_inputs = {ort_session.get_inputs()[0].name: prepared_batch}
            onnx_results = ort_session.run(None, ort_inputs)  # infer with ONNX

            # ~ Finally, we persist the results of inference.
            # For now, collated_batch will always have an "object_id" key that
            # is a list of strings. However, we should move to a state where the
            # object ids are taken from the primary dataset's "primary_id_field",
            # which will contain the required data - then remove the "object_id" key.
            if "object_id" not in collated_batch:
                msg = "Dataset dictionary should be returning object_ids to avoid ordering errors. "
                msg = f"Could not determine object IDs from batch. Batch has keys {collated_batch.keys()}"
                raise RuntimeError(msg)

            # ~ We may not need to do the list comprehension for batch_results, it's
            # possible that ONNX will already return it in this form.
            self.results_writer.write_batch(collated_batch["object_id"], [t for t in onnx_results])

        self.results_writer.write_index()
