import logging

import torch
from numpy import typing as npt

from hyrax.context import get_context, use_context
from hyrax.pytorch_ignite import results_to_numpy

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

    def run(self) -> "InferStreamSession":
        """Set up the model and return a session for streaming inference.

        There are two ways to drive the session:

        1. **Data-source driven** — configure a streaming dataset under
           ``[data_request.infer_stream]`` (e.g. ``KafkaStreamDataset``). A DataLoader is
           built over it, so the returned session can be iterated directly::

               with hy.infer_stream() as session:
                   for batch, results in session:
                       ...

        2. **Manual** — leave ``[data_request.infer_stream]`` unset and feed batches
           yourself::

               with hy.infer_stream() as session:
                   results = session.process(batch)

        Returns
        -------
        InferStreamSession
            A context manager / session object. Iterate it (data-source driven) or call
            ``session.process(batch)`` (manual); call ``session.close()`` when done.
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
        )
        from hyrax.tensorboardx_logger import close_tensorboard_logger, init_tensorboard_logger

        config = self.config

        # Create a timestamped results directory. This happens before the model is built
        # so that the run context is populated by the time the model's __init__ runs,
        # matching the ordering used by the train, infer and test verbs.
        results_dir = create_results_dir(config, "infer_stream")

        model = setup_model(config)

        # A configured [data_request.infer_stream] group makes the session iterable over
        # that stream. Without one the session is manual: the caller feeds process().
        provider = None
        data_loader = None
        if config.get("data_request"):
            provider = setup_dataset(config, splits=InferStream.REQUIRED_DATA_GROUPS).get("infer_stream")
        if provider is not None:
            data_loader = dist_data_loader(provider, config)

        # set model in eval mode
        model.eval()

        # Start TensorBoard logger
        init_tensorboard_logger(log_dir=results_dir)

        log_runtime_config(config, results_dir)

        # load weights, save the model and place the model on the correct device.
        load_model_weights(config, model, "infer_stream")
        model.save(results_dir / "inference_weights.pth")
        model = auto_model(model)

        logger.info(f"Saving infer_stream results at: {results_dir}")

        # Build the per-batch process function (same partial used by create_engine)
        device = idist_device()
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

        # A session keeps running model code after infer_stream's run() has
        # returned and released its context. Hold that context so process() can
        # re-install it, otherwise a model reading get_context() during streaming
        # would find nothing.
        self._context = get_context()

    def __iter__(self):
        """Iterate the configured data source, processing each batch as it arrives.

        Yields
        ------
        tuple[dict, np.ndarray | dict[str, np.ndarray]]
            The collated input ``batch`` and the model ``results`` for it, in the form
            returned by :meth:`process`.

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

    def _close_provider(self):
        """Tear down the streaming data source, releasing any connection it holds.

        Kept separate from :meth:`stop` because stopping is only a signal: a data loader
        left suspended mid-iteration (the caller hit an exception, or broke out of the
        loop) never resumes to clean up after itself, so the connection has to be closed
        from the outside or it lingers for the life of the process.
        """
        if self._provider is None:
            return
        close = getattr(self._provider, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as err:
            # Teardown of the data source must not prevent the result commit below.
            logger.warning(f"Error closing the streaming data source: {err}")

    def process(self, batch: dict) -> npt.NDArray | dict[str, npt.NDArray]:
        """Run inference on a single batch and save results.

        Parameters
        ----------
        batch : dict
            Must contain ``"object_id"`` (list of str) and model-specific data fields.

        Returns
        -------
        np.ndarray | dict[str, np.ndarray]
            Model output on CPU, detached from the computation graph. When the model's
            ``infer_batch`` returns a dict of tensors, this is a dict with the same keys
            whose values are numpy arrays; otherwise it is a single numpy array.

        Raises
        ------
        RuntimeError
            If the session has already been closed, or if the model returns something
            other than a tensor or a dict of tensors.
        """
        if self._closed:
            raise RuntimeError("InferStreamSession is closed. Cannot call process() after close().")

        # `use_context(...)` is here because we want model methods,
        # i.e. the method assigned to self._process_func, to have access to the
        # verb context that was established in `infer_stream.run`. `use_context`
        # ensures that the correct context is active during inference.
        with use_context(self._context), torch.no_grad():
            result = self._process_func(None, batch)

            if self._config["infer_stream"]["save_model_output"]:
                self._save_batch(batch, result)

        return results_to_numpy(result)

    def close(self):
        """Commit results and return the result dataset.

        Returns
        -------
        ResultDataset
            The accumulated results from all batches processed in this session.
        """
        if self._closed:
            if self._config["infer_stream"]["save_model_output"]:
                return self._load_dataset(self._config, self._results_dir)
            return None
        # End any in-progress streaming iteration, then release the data source itself.
        # __exit__ calls close() on every exit path, so this is what makes
        # `with hy.infer_stream() as session:` a real guarantee that the stream's
        # connection is dropped even when the body raises.
        self.stop()
        self._close_provider()
        self.data_loader = None

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
