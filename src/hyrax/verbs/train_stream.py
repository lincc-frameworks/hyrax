import logging

from .verb_registry import Verb, hyrax_verb

logger = logging.getLogger(__name__)


@hyrax_verb
class TrainStream(Verb):
    """Streaming training verb — loads the model once, trains on batches on demand."""

    cli_name = "train_stream"
    add_parser_kwargs = {}
    description = "Run streaming training: load the model once and train on batches interactively."

    REQUIRED_DATA_GROUPS = ("train_stream",)
    OPTIONAL_DATA_GROUPS = ()

    @staticmethod
    def setup_parser(parser):
        """No CLI arguments needed."""
        pass

    def run_cli(self, args=None):
        """CLI stub — train_stream is a programmatic API only."""
        raise NotImplementedError(
            "train_stream is a programmatic API; use hyrax.train_stream() in Python/notebook."
        )

    def run(self, sample_batch: dict | None = None) -> "TrainStreamSession":
        """Set up the model and return a session for streaming training.

        There are two ways to drive the session:

        1. **Data-source driven** (``sample_batch=None``) — configure a streaming
           dataset under ``[data_request.train_stream]`` (e.g. ``KafkaStreamDataset``).
           The model is pre-flighted from the stream itself and a DataLoader is built,
           so the returned session can be iterated directly::

               with hy.train_stream() as session:
                   for batch, metrics in session:
                       print(metrics["loss"])

        2. **Manual** — pass a representative ``sample_batch`` and feed batches yourself::

               with hy.train_stream(sample_batch=batch) as session:
                   metrics = session.train_batch(batch)

        Because a live stream is open-ended (no epochs, no length, no train/validate/test
        splits), this verb does **not** use the Ignite ``create_trainer`` /
        ``trainer.run(..., max_epochs=...)`` path. Instead it reuses the same per-batch
        function the trainer's engine calls (:func:`~hyrax.pytorch_ignite.create_process_func`
        with ``"train_batch"``), which already runs the model's self-contained optimizer
        step, and drives it from the session loop.

        Parameters
        ----------
        sample_batch : dict | None
            A representative batch dict with ``"object_id"`` and model-specific data
            fields, used to pre-flight the model architecture. When ``None``, the model
            is pre-flighted from a ``[data_request.train_stream]`` streaming dataset
            instead.

        Returns
        -------
        TrainStreamSession
            A context manager / session object. Iterate it (data-source driven) or call
            ``session.train_batch(batch)`` (manual); call ``session.close()`` when done to
            persist the final weights and get the trained model back.

        Raises
        ------
        ValueError
            If ``sample_batch`` is None and no ``[data_request.train_stream]`` is configured.
        """
        from pathlib import Path

        import mlflow
        from ignite.distributed import auto_model
        from ignite.distributed import device as idist_device

        from hyrax.config_utils import create_results_dir, log_runtime_config
        from hyrax.pytorch_ignite import (
            create_process_func,
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
                    "train_stream requires either a `sample_batch` argument or a "
                    "[data_request.train_stream] configuration to build the data source."
                )
            datasets = setup_dataset(config, splits=TrainStream.REQUIRED_DATA_GROUPS)
            provider = datasets.get("train_stream")
            if provider is None:
                raise ValueError(
                    "No [data_request.train_stream] group found. Configure it with a "
                    "streaming dataset_class, or pass an explicit `sample_batch`."
                )
            # Pre-flight the model from the stream (peeks one sample without losing it).
            model = setup_model(config, provider)
            data_loader = dist_data_loader(provider, config)
        else:
            model = setup_model_from_sample(config, sample_batch)

        # Put the model in training mode.
        model.train()

        # If a warm-start weights file is specified, load it before wrapping the model with
        # idist.auto_model (the distributed wrapper) to avoid parameter key name mismatches.
        if config["train_stream"]["model_weights_file"]:
            from hyrax.models.model_utils import load_model_weights

            load_model_weights(config, model, "train_stream")
            logger.info(f"Loaded warm-start weights: {config['train_stream']['model_weights_file']}")

        # Create a timestamped results directory and start logging.
        results_dir = create_results_dir(config, "train_stream")
        init_tensorboard_logger(log_dir=results_dir)
        log_runtime_config(config, results_dir)

        logger.info(f"Saving train_stream results at: {results_dir}")

        # Wrap for (possibly distributed) execution, but keep the unwrapped model around so
        # we can call `.save()` and reference `.optimizer`. On a single device auto_model is
        # a no-op, so both names refer to the same object.
        wrapped_model = auto_model(model)
        device = idist_device()
        process_func = create_process_func("train_batch", device, wrapped_model, config)

        # Start an MLflow run that spans the whole session. A stream has no fixed end, so we
        # cannot wrap the training loop in a `with mlflow.start_run()` block the way batch
        # train does; the run stays open across the session and is ended in `close()`.
        results_root_dir = Path(config["general"]["results_dir"]).expanduser().resolve()
        (results_root_dir / "mlflow").mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + str(results_root_dir / "mlflow" / "mlflow.db"))
        mlflow.set_experiment(str(config["train_stream"]["experiment_name"]))
        run_name = (
            str(config["train_stream"]["run_name"])
            if config["train_stream"]["run_name"]
            else results_dir.name
        )
        mlflow.start_run(log_system_metrics=True, run_name=run_name)
        TrainStream._log_params(config, results_dir)

        return TrainStreamSession(
            process_func,
            model,
            config,
            results_dir,
            close_tensorboard_logger,
            data_loader=data_loader,
            provider=provider,
        )

    @staticmethod
    def _log_params(config, results_dir):
        """Log static run parameters to MLflow (mirrors ``Train._log_params``)."""
        import mlflow

        mlflow.log_param("Results Directory", results_dir)
        mlflow.log_params(config["model"])
        mlflow.log_param("batch_size", config["data_loader"]["batch_size"])

        criterion_name = config["criterion"]["name"]
        mlflow.log_param("criterion", criterion_name)
        if criterion_name in config:
            mlflow.log_params(config[criterion_name])

        optimizer_name = config["optimizer"]["name"]
        mlflow.log_param("optimizer", optimizer_name)
        if optimizer_name in config:
            mlflow.log_params(config[optimizer_name])


class TrainStreamSession:
    """Context manager for streaming training.

    Holds a loaded model and its optimizer. When constructed with a ``data_loader``
    (the data-source-driven path), the session is iterable and yields ``(batch, metrics)``
    pairs as data arrives; otherwise feed batches yourself with :meth:`train_batch`.

    Each processed batch runs the model's self-contained training step (``zero_grad`` →
    forward → loss → ``backward`` → ``optimizer.step``) and returns its metrics dict.
    Model weights are persisted periodically (``save_weights_every``) and always on
    :meth:`close`.

    .. warning::
        :meth:`train_batch` / :meth:`process` are **not** thread-safe. Do not call them
        concurrently.
    """

    def __init__(
        self,
        process_func,
        model,
        config,
        results_dir,
        close_logger_fn,
        data_loader=None,
        provider=None,
    ):
        from hyrax.tensorboardx_logger import get_tensorboard_logger

        self._process_func = process_func
        self._model = model
        self._config = config
        self._results_dir = results_dir
        self._close_logger = close_logger_fn
        self.data_loader = data_loader
        self._provider = provider
        self._closed = False
        self._batch_count = 0
        self._tb_logger = get_tensorboard_logger()

    def __iter__(self):
        """Iterate the configured data source, training on each batch as it arrives.

        Yields
        ------
        tuple[dict, dict | None]
            The collated input ``batch`` and the training ``metrics`` for it (or ``None``
            if the batch was skipped for being empty / smaller than ``min_batch_size``).

        Raises
        ------
        RuntimeError
            If the session was created without a data source (no
            ``[data_request.train_stream]`` configuration).
        """
        if self.data_loader is None:
            raise RuntimeError(
                "This TrainStreamSession has no data source to iterate. Configure "
                "[data_request.train_stream], or feed batches with train_batch(batch)."
            )
        for batch in self.data_loader:
            yield batch, self.process(batch)

    def _batch_num_samples(self, batch) -> int | None:
        """Return the number of samples in a collated batch, or ``None`` if unknown."""
        if isinstance(batch, dict):
            object_id = batch.get("object_id")
            if object_id is not None:
                return len(object_id)
        return None

    def process(self, batch: dict) -> dict | None:
        """Run one training step on a single batch.

        The batch is passed through the model's ``train_batch`` (which performs the full
        optimizer step) unless it is empty or smaller than the configured
        ``min_batch_size``, in which case it is skipped.

        Parameters
        ----------
        batch : dict
            Must contain ``"object_id"`` (list of str) and model-specific data fields.

        Returns
        -------
        dict | None
            The training metrics (e.g. ``{"loss": ...}``), or ``None`` if the batch was
            skipped.

        Raises
        ------
        RuntimeError
            If the session has already been closed.
        """
        if self._closed:
            raise RuntimeError("TrainStreamSession is closed. Cannot call process() after close().")

        # Skip empty / too-small batches. Streaming sources emit ragged batches (a partial
        # batch is flushed when the latency timeout fires), and a size-1 batch can break
        # batch-size-sensitive layers (e.g. BatchNorm) in some user models.
        num_samples = self._batch_num_samples(batch)
        if num_samples is not None:
            if num_samples == 0:
                logger.debug("Skipping empty batch.")
                return None
            min_batch_size = self._config["train_stream"]["min_batch_size"]
            if min_batch_size and num_samples < min_batch_size:
                logger.debug(f"Skipping batch of {num_samples} < min_batch_size ({min_batch_size}).")
                return None

        # Gradients are required here (unlike inference), so no torch.no_grad().
        result = self._process_func(None, batch)
        self._batch_count += 1

        self._log_metrics(result)

        save_every = self._config["train_stream"]["save_weights_every"]
        if save_every and self._batch_count % save_every == 0:
            self.save_weights()

        return result

    # `train_batch` reads better than `process` for the training API; keep both so the
    # session mirrors InferStreamSession's `process` while offering a training-native name.
    train_batch = process

    def _log_metrics(self, result) -> None:
        """Log per-batch metrics to TensorBoard and (if a run is active) MLflow."""
        if not isinstance(result, dict):
            return
        import mlflow

        active = mlflow.active_run() is not None
        for metric, value in result.items():
            self._tb_logger.add_scalar(f"training/training/{metric}", value, self._batch_count)
            if active:
                mlflow.log_metrics({f"training/{metric}": value}, step=self._batch_count)

    def save_weights(self) -> None:
        """Persist the current model weights to the results directory."""
        self._model.save(self._results_dir / self._config["train_stream"]["weights_filename"])
        logger.debug(f"Saved weights after {self._batch_count} batches.")

    def stop(self):
        """Signal the underlying streaming data source to stop iterating."""
        if self._provider is not None and hasattr(self._provider, "stop"):
            self._provider.stop()

    def close(self):
        """Persist final weights, end logging, and return the trained model.

        Returns
        -------
        torch.nn.Module
            The trained model.
        """
        if self._closed:
            return self._model

        # End any in-progress streaming iteration before tearing down.
        self.stop()
        self.save_weights()
        self._closed = True

        import mlflow

        if mlflow.active_run() is not None:
            mlflow.end_run()

        self._close_logger()
        logger.info("TrainStream session closed.")
        return self._model

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False
