import logging

import torch
from numpy import typing as npt

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class InferStream(Verb):
    """Streaming inference verb — loads model once, processes batches on demand."""

    cli_name = "infer_stream"
    add_parser_kwargs = {}
    description = "Run streaming inference: load model once and process batches interactively."

    REQUIRED_DATA_GROUPS = ("infer_stream",)
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

        There are two ways to drive the session:

        1. **Data-source driven** (``sample_batch=None``) — configure a streaming
           dataset under ``[data_request.infer_stream]`` (e.g. ``KafkaStreamDataset``).
           The model is pre-flighted from the stream itself and a DataLoader is built,
           so the returned session can be iterated directly::

               with hy.infer_stream() as session:
                   for batch, results in session:
                       ...

        2. **Manual** — pass a representative ``sample_batch`` and feed batches yourself::

               with hy.infer_stream(sample_batch=batch) as session:
                   results = session.process(batch)

        Parameters
        ----------
        sample_batch : dict | None
            A representative batch dict with ``"object_id"`` and model-specific data
            fields, used to pre-flight the model architecture. When ``None``, the model
            is pre-flighted from a ``[data_request.infer_stream]`` streaming dataset
            instead.

        Returns
        -------
        InferStreamSession
            A context manager / session object. Iterate it (data-source driven) or call
            ``session.process(batch)`` (manual); call ``session.close()`` when done.

        Raises
        ------
        ValueError
            If ``sample_batch`` is None and no ``[data_request.infer_stream]`` is configured.
        """
        from ignite.distributed import auto_model
        from ignite.distributed import device as idist_device

        from hyrax.config_utils import create_results_dir, log_runtime_config
        from hyrax.datasets.result_factories import load_results_dataset
        from hyrax.models.model_utils import load_model_weights
        from hyrax.pytorch_ignite import (
            create_process_func,
            create_save_batch_callback,
            dist_data_loader,
            setup_dataset,
            setup_model,
            setup_model_from_sample,
        )
        from hyrax.tensorboardx_logger import close_tensorboard_logger, init_tensorboard_logger

        config = self.config

        # Build the model either from a configured streaming dataset (preferred, enables
        # session iteration) or from an explicitly supplied sample batch.
        provider = None
        data_loader = None
        if sample_batch is None:
            if not config.get("data_request"):
                raise ValueError(
                    "infer_stream requires either a `sample_batch` argument or a "
                    "[data_request.infer_stream] configuration to build the data source."
                )
            datasets = setup_dataset(config, splits=InferStream.REQUIRED_DATA_GROUPS)
            provider = datasets.get("infer_stream")
            if provider is None:
                raise ValueError(
                    "No [data_request.infer_stream] group found. Configure it with a "
                    "streaming dataset_class, or pass an explicit `sample_batch`."
                )
            # Pre-flight the model from the stream (peeks one sample without losing it).
            model = setup_model(config, provider)
            data_loader = dist_data_loader(provider, config)
        else:
            model = setup_model_from_sample(config, sample_batch)

        # set model in eval mode
        model.eval()

        # Create a timestamped results directory
        results_dir = create_results_dir(config, "infer_stream")

        # Start TensorBoard logger
        init_tensorboard_logger(log_dir=results_dir)

        log_runtime_config(config, results_dir)

        # load weights, save the model and place the model on the correct device.
        load_model_weights(config, model, "infer_stream")
        model.save(results_dir / "inference_weights.pth")
        model = auto_model(model)

        logger.info(f"Saving infer_stream results at: {results_dir}")

        # Build the per-batch process function (same partial used by create_engine)
        device = idist_device
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
            data_loader=data_loader,
            provider=provider,
        )


class InferStreamSession:
    """Context manager for streaming inference.

    Holds a loaded model and Lance writer. When constructed with a ``data_loader``
    (the data-source-driven path), the session is iterable and yields
    ``(batch, results)`` pairs as data arrives; otherwise feed batches yourself with
    :meth:`process`.

    .. warning::
        ``process()`` is **not** thread-safe. Do not call it concurrently.
    """

    def __init__(
        self,
        process_func,
        save_batch_callback,
        config,
        results_dir,
        close_logger_fn,
        load_dataset_fn,
        data_loader=None,
        provider=None,
    ):
        self._process_func = process_func
        self._save_batch = save_batch_callback
        self._config = config
        self._results_dir = results_dir
        self._close_logger = close_logger_fn
        self._load_dataset = load_dataset_fn
        self.data_loader = data_loader
        self._provider = provider
        self._closed = False

    def __iter__(self):
        """Iterate the configured data source, processing each batch as it arrives.

        Yields
        ------
        tuple[dict, np.ndarray]
            The collated input ``batch`` and the model ``results`` for it.

        Raises
        ------
        RuntimeError
            If the session was created without a data source (no
            ``[data_request.infer_stream]`` configuration).
        """
        if self.data_loader is None:
            raise RuntimeError(
                "This InferStreamSession has no data source to iterate. Configure "
                "[data_request.infer_stream], or feed batches with process(batch)."
            )
        for batch in self.data_loader:
            yield batch, self.process(batch)

    def stop(self):
        """Signal the underlying streaming data source to stop iterating."""
        if self._provider is not None and hasattr(self._provider, "stop"):
            self._provider.stop()

    def process(self, batch: dict) -> npt.NDArray:
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

        # End any in-progress streaming iteration before tearing down.
        self.stop()

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
