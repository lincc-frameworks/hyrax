import logging

import torch

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class InferStream(Verb):
    """Streaming inference verb — loads model once, processes batches on demand."""

    cli_name = "infer_stream"
    add_parser_kwargs = {}
    description = "Run streaming inference: load model once and process batches interactively."

    REQUIRED_DATA_GROUPS = ()
    OPTIONAL_DATA_GROUPS = ()

    @staticmethod
    def setup_parser(parser):
        """No CLI arguments needed."""
        pass

    def run_cli(self, args=None):
        """CLI stub — infer_stream is a programmatic API only."""
        raise NotImplementedError(
            "infer_stream is a programmatic API; use hyrax.infer_stream() in Python/notebook."
        )

    def run(self, sample_batch: dict | None = None) -> "InferStreamSession":
        """Set up the model and return a session for streaming inference.

        Parameters
        ----------
        sample_batch : dict | None
            A representative batch dict with ``"object_id"`` and ``"data"`` keys.
            Used to pre-flight the model architecture. Required.

        Returns
        -------
        InferStreamSession
            A context manager / session object. Call ``session.process(batch)``
            for each batch and ``session.close()`` when done.

        Raises
        ------
        ValueError
            If ``sample_batch`` is None.
        """
        if sample_batch is None:
            raise ValueError(
                "sample_batch is required for infer_stream. "
                "Pass a representative batch dict with 'object_id' and 'data' keys."
            )

        from ignite.distributed import auto_model
        from ignite.distributed import device as idist_device

        from hyrax.config_utils import create_results_dir, log_runtime_config
        from hyrax.datasets.result_factories import load_results_dataset
        from hyrax.models.model_utils import load_model_weights
        from hyrax.pytorch_ignite import (
            create_process_func,
            create_save_batch_callback,
            setup_model_from_sample,
        )
        from hyrax.tensorboardx_logger import close_tensorboard_logger, init_tensorboard_logger

        config = self.config

        # Create a timestamped results directory
        results_dir = create_results_dir(config, "infer_stream")

        # Start TensorBoard logger
        init_tensorboard_logger(log_dir=results_dir)

        # Build model from the representative sample batch
        model = setup_model_from_sample(config, sample_batch)
        model.eval()

        device = idist_device()
        # torch.set_default_device(device.type)  # TODO: I don't think this line is needed
        model = auto_model(model)

        load_model_weights(config, model, "infer_stream")
        log_runtime_config(config, results_dir)
        model.save(results_dir / "inference_weights.pth")

        logger.info(f"Saving infer_stream results at: {results_dir}")

        # Build the per-batch process function (same partial used by create_engine)
        process_func = create_process_func("infer_batch", device, model, config)

        # Create the Lance writer callback (reused across all .process() calls)
        save_batch_callback = create_save_batch_callback(results_dir)

        return InferStreamSession(
            process_func,
            save_batch_callback,
            config,
            results_dir,
            close_tensorboard_logger,
            load_results_dataset,
        )


class InferStreamSession:
    """Context manager for streaming inference.

    Holds a loaded model and Lance writer; accepts batches one at a time.

    .. warning::
        ``process()`` is **not** thread-safe. Do not call it concurrently.
    """

    def __init__(
        self, process_func, save_batch_callback, config, results_dir, close_logger_fn, load_dataset_fn
    ):
        self._process_func = process_func
        self._save_batch = save_batch_callback
        self._config = config
        self._results_dir = results_dir
        self._close_logger = close_logger_fn
        self._load_dataset = load_dataset_fn
        self._closed = False

    def process(self, batch: dict) -> torch.Tensor:
        """Run inference on a single batch and save results.

        Parameters
        ----------
        batch : dict
            Must contain ``"object_id"`` (list of str) and model-specific data fields.

        Returns
        -------
        np.ndarray
            Model output on CPU, detached from the computation graph.

        Raises
        ------
        RuntimeError
            If the session has already been closed.
        """
        if self._closed:
            raise RuntimeError("InferStreamSession is closed. Cannot call process() after close().")

        with torch.no_grad():
            result = self._process_func(None, batch)

        if self._config["infer_stream"]["save_model_output"]:
            self._save_batch(batch, result)
        return result.detach().cpu().numpy()

    def close(self):
        """Commit results and return the result dataset.

        Returns
        -------
        ResultDataset
            The accumulated results from all batches processed in this session.
        """
        if self._closed:
            return self._load_dataset(self._config, self._results_dir)

        if self._config["infer_stream"]["save_model_output"]:
            self._save_batch.data_writer.commit()
        self._closed = True
        self._close_logger()
        logger.info("InferStream session closed.")

        if self._config["infer_stream"]["save_model_output"]:
            return self._load_dataset(self._config, self._results_dir)
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False
