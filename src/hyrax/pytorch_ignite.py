import functools
import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ignite.distributed as idist
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    import mlflow

from collections.abc import Iterator, Sequence

import torch
from ignite.engine import Engine, EventEnum, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.tqdm_logger import ProgressBar
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, WeightedRandomSampler

from hyrax.datasets.data_provider import DataProvider, generate_data_request_from_config
from hyrax.models.model_registry import fetch_model_class
from hyrax.tensorboardx_logger import get_tensorboard_logger
from hyrax.trace import get_trace

logger = logging.getLogger(__name__)

_LEGACY_SPLIT_KEYS = ("train_size", "validate_size", "test_size")


class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Args:
        indices : sequence
            a sequence of indices
    """

    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in self.indices:
            yield i

    def __len__(self) -> int:
        return len(self.indices)


def setup_dataset(
    config: dict,
    *,
    splits: tuple[str, ...] | None = None,
    shuffle: bool = True,
) -> dict[str, DataProvider]:
    """Create DataProvider instances for each requested data group.

    Parameters
    ----------
    config : dict
        The runtime configuration.
    splits : tuple[str, ...] | None, optional
        When provided, only create DataProvider instances for the listed groups.
        When ``None`` every group in the data_request is loaded.
    shuffle : bool, optional
        Unused; kept for backward-compatibility with call sites that still pass
        it.  Split shuffling is now handled by ``splitting_utils.create_splits``.

    Returns
    -------
    dict[str, DataProvider]
        Mapping of data group names to DataProvider instances.
    """

    found = [k for k in _LEGACY_SPLIT_KEYS if k in config.get("data_set", {})]
    if found:
        raise RuntimeError(
            f"Legacy split configuration keys found in [data_set]: {found}\n\n"
            "The train_size/validate_size/test_size configuration style has been removed.\n"
            "Please migrate your split configuration to [split].\n\n"
            "Example:\n"
            "  [data_request.train.data]\n"
            "  dataset_class = 'YourDataset'\n"
            "  data_location = '/path/to/data'\n"
            "  primary_id_field = 'id'\n\n"
            "  [data_request.validate.data]\n"
            "  dataset_class = 'YourDataset'\n"
            "  data_location = '/path/to/data'\n"
            "  primary_id_field = 'id'\n\n"
            "  [split]\n"
            "  train = 0.8\n"
            "  validate = 0.2\n\n"
            "For more information, see: https://hyrax.readthedocs.io/en/stable/dataset_splits.html"
        )

    data_request = generate_data_request_from_config(config)
    keys = splits if splits is not None else tuple(data_request.keys())
    return {k: DataProvider(config, data_request[k]) for k in keys if k in data_request}


def setup_model(config: dict, dataset: DataProvider) -> torch.nn.Module:
    """Create a model object based on the configuration.

    Parameters
    ----------
    config : dict
        The runtime configuration
    dataset : DataProvider
        The dataset object that will provide data to the model for training or
        inference. Here it is only used to provide a data sample to the model so
        that it can resize itself at runtime if necessary.

    Returns
    -------
    torch.nn.Module
        An instance of the model class specified in the configuration
    """
    from hyrax.trace import reset_trace

    # Fetch model class specified in config and create an instance of it
    model_cls = fetch_model_class(config)

    # Grab a single data sample
    data_sample = dataset.sample_data()

    # Collate the data sample
    collated_sample = dataset.collate([data_sample])

    # Prepare the data sample with the model's prepare_inputs function
    prepared_sample = model_cls.prepare_inputs(collated_sample)

    # Provide the sample for runtime modifications to the model architecture
    retval = model_cls(config=config, data_sample=prepared_sample)  # type: ignore[attr-defined]

    # After model pre-flighting succeeds (presumably) reset the trace so it represents
    # just what the verb does afterward.
    reset_trace()

    return retval


def dist_data_loader(
    dataset: Dataset,
    config: dict,
    shuffle: bool = False,
) -> DataLoader:
    """Create Pytorch Ignite distributed data loaders.

    It is recommended that each verb needing dataloaders only call this function once.

    Parameters
    ----------
    dataset : hyrax.datasets.dataset_registry.HyraxDataset
        A Hyrax dataset instance.  When *dataset* is a :class:`DataProvider`
        with ``split_indices`` set (by :func:`~hyrax.splitting_utils.create_splits`),
        the loader is restricted to those indices via a :class:`~torch.utils.data.Subset`.
        When ``split_weights`` is also set, a
        :class:`~torch.utils.data.WeightedRandomSampler` is used so that
        under-represented classes are over-sampled to achieve the configured
        class distribution.
    config : dict
        Hyrax runtime configuration
    shuffle : bool, optional
        If ``True`` and no weights are present, a
        :class:`~torch.utils.data.SubsetRandomSampler` is used for uniform
        shuffling.  If ``False`` and no weights, a sequential sampler preserves
        deterministic order.  Ignored when ``split_weights`` is set (weighted
        sampling always draws with replacement).  Defaults to ``False`` so
        non-training verbs preserve deterministic order.

    Returns
    -------
    DataLoader
        The distributed dataloader.
    """

    # Extract the config dictionary that will be provided as kwargs to the DataLoader.
    # Hyrax controls ordering through explicit samplers; warn and ignore legacy
    # ``data_loader.shuffle`` if an old/unversioned config still contains it.
    data_loader_kwargs = dict(config["data_loader"])
    if "shuffle" in data_loader_kwargs:
        msg = (
            "config['data_loader']['shuffle'] is ignored and is not passed to PyTorch DataLoader. "
            "Hyrax controls dataloader ordering with explicit samplers; use config['train']['shuffle'] "
            "to control training sample shuffling to support reproducibility."
        )
        logger.warning(msg)
        data_loader_kwargs.pop("shuffle")

    # TODO: Actually DataProvider.collate. Callsites and parameter signature above have not been updated.
    data_loader_kwargs["collate_fn"] = dataset.collate

    torch_rng = torch.Generator()

    seed = config["data_set"]["seed"] if config["data_set"]["seed"] else None
    if seed is not None:
        torch_rng.manual_seed(seed)

    indexes = list(range(len(dataset)))
    weights = None
    if isinstance(dataset, DataProvider) and dataset.split_indices is not None:
        indexes = dataset.split_indices
        weights = dataset.split_weights

    sub_dataset = Subset(dataset, indexes)
    n = len(indexes)

    # If no weights come from the split, then substitute with the correct number of 1's
    if weights is None:
        weights = np.ones(n, dtype=np.float32)

    # If we are in trace mode, the sampler should stop after the batch size
    # since the batch size has been set to the number of rows the user wanted to trace
    if get_trace():
        trace_limit = data_loader_kwargs.get("batch_size", 1)
        if shuffle:
            # Limit via WeightedRandomSampler
            n = trace_limit
        else:
            # Limit via Subset
            sub_dataset = Subset(sub_dataset, list(range(trace_limit)))

    sampler = (
        WeightedRandomSampler(weights=weights, num_samples=n, generator=torch_rng, replacement=True)
        if shuffle
        else None
    )

    return idist.auto_dataloader(sub_dataset, sampler=sampler, **data_loader_kwargs)


# TODO: Clean up the input variables here.
def _inner_loop(func, prepare_inputs, device, config, engine, batch):
    """This wraps a model-specific function (func) to move data to the appropriate device."""
    # Pass the collated batch through the model's prepare_inputs function
    batch = prepare_inputs(batch)

    # Convert the data to numpy and place it on the device explicitly.
    # This allows us to control when the tensor makes it on to the device without setting
    # torch.default_device. Thus user code will default to making 'cpu' tensors unless the user
    # explicitly specifies a different device.
    #
    # The hope is that even in the presence of user code in datasets that might manipulate tensors
    # with torch primitives, functionally all of the tensors get clocked out to the GPU by this
    # line of code.
    #
    # We use torch.from_numpy() over torch.tensor() to avoid the copy of data that occurs in the latter.

    if isinstance(batch, tuple):
        batch = tuple(torch.from_numpy(i).to(device) if i is not None else None for i in batch)
    elif batch is not None:
        batch = torch.from_numpy(batch).to(device)

    return func(batch)


def _create_process_func(funcname, device, model, config):
    inner_step = extract_model_method(model, funcname)
    prepare_inputs = extract_model_method(model, "prepare_inputs")
    inner_loop = functools.partial(_inner_loop, inner_step, prepare_inputs, device, config)
    return inner_loop


def create_engine(funcname: str, device: torch.device, model: torch.nn.Module, config: dict) -> Engine:
    """Unified creation of the pytorch engine object for either an evaluator or trainer.

    This function will automatically unwrap a distributed model to find the necessary function, and construct
    the necessary functions to transfer data to the device on every batch, so model code can be the same no
    matter where the model is being run.

    Parameters
    ----------
    funcname : str
        The function name on the model that we will call in the core of the engine loop, and be called once
        per batch
    device : torch.device
        The device the engine will run the model on
    model : torch.nn.Module
        The Model the engine will be using
    config : dict
        The runtime config in use
    """
    return Engine(_create_process_func(funcname, device, model, config))


def extract_model_method(model, method_name):
    """Extract a method from a model, which may be wrapped in a DistributedDataParallel
    or DataParallel object. For instance, method_name could be `train_batch` or
    `infer_batch`.

    Parameters
    ----------
    model : nn.Module, DistributedDataParallel, or DataParallel
        The model to extract the method from
    method_name : str
        Name of the method to extract

    Returns
    -------
    Callable
        The method extracted from the model
    """
    wrapped = type(model) is DistributedDataParallel or type(model) is DataParallel

    # Check to see if the model has the requested method
    if not hasattr(model.module if wrapped else model, method_name):
        raise RuntimeError(f"Model does not have required method: {method_name}")

    return getattr(model.module if wrapped else model, method_name)


def create_evaluator(
    model: torch.nn.Module, save_function: Callable[[torch.Tensor, torch.Tensor], Any], config: dict
) -> Engine:
    """Creates an evaluator engine
    Primary purpose of this function is to attach the appropriate handlers to an evaluator engine

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate

    save_function : Callable[[torch.Tensor], Any]
        A function which will receive Engine.state.output at the end of each iteration. The intent
        is for the results of evaluation to be saved.

    config : dict
        The runtime config in use

    Returns
    -------
    pytorch-ignite.Engine
        Engine object which when run will evaluate the model.
    """
    device = idist.device()
    model.eval()
    wrapped_model = idist.auto_model(model)
    evaluator = create_engine("infer_batch", device, wrapped_model, config)

    @evaluator.on(Events.STARTED)
    def log_eval_start(evaluator):
        logger.debug(f"Evaluating model on device: {device}")
        logger.debug(f"Total epochs: {evaluator.state.max_epochs}")

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_iteration_complete(evaluator):
        save_function(evaluator.state.batch, evaluator.state.output)

    @evaluator.on(Events.COMPLETED)
    def log_total_time(evaluator):
        logger.info(f"Total evaluation time: {evaluator.state.times['COMPLETED']:.2f}[s]")

    pbar = ProgressBar(persist=False, bar_format="")
    pbar.attach(evaluator)

    evaluator.hyrax_label = "evaluator"
    return evaluator


#! There will likely be a significant amount of code duplication between the
#! `create_trainer` and `create_validator` functions. We should find a way to
#! refactor this code to reduce duplication.
def create_validator(
    model: torch.nn.Module,
    config: dict,
    validation_data_loader: DataLoader,
    trainer: Engine,
) -> Engine:
    """This function creates a Pytorch Ignite engine object that will be used to
    validate the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    config : dict
        Hyrax runtime configuration
    validation_data_loader : DataLoader
        The data loader for the validation data
    trainer : pytorch-ignite.Engine
        The engine object that will be used to train the model. We will use specific
        hooks in the trainer to determine when to run the validation engine.

    Returns
    -------
    pytorch-ignite.Engine
        Engine object that will be used to train the model.
    """

    device = idist.device()
    wrapped_model = idist.auto_model(model)
    tensorboardx_logger = get_tensorboard_logger()

    validator = create_engine("validate_batch", device, wrapped_model, config)
    fixup_engine(validator)

    @validator.on(Events.STARTED)
    def set_model_to_eval_mode():
        wrapped_model.eval()

    @validator.on(Events.COMPLETED)
    def set_model_to_train_mode():
        wrapped_model.train()

    @validator.on(HyraxEvents.HYRAX_EPOCH_COMPLETED)
    def log_training_loss():
        logger.debug(f"Validation run time: {validator.state.times['EPOCH_COMPLETED']:.2f}[s]")
        logger.debug(f"Validation metrics: {validator.state.output}")
        model.final_validation_metrics = validator.state.output

    @trainer.on(HyraxEvents.HYRAX_EPOCH_COMPLETED)
    def run_validation():
        with torch.no_grad():
            validator.run(validation_data_loader)

    def log_validation_loss(validator, trainer):
        step = trainer.state.get_event_attrib_value(Events.EPOCH_COMPLETED)
        for m in trainer.state.output:
            tensorboardx_logger.add_scalar(f"training/validation/{m}", validator.state.output[m], step)
            mlflow.log_metrics({f"validation/{m}": validator.state.output[m]}, step=step)

    validator.add_event_handler(HyraxEvents.HYRAX_EPOCH_COMPLETED, log_validation_loss, trainer)

    validator.hyrax_label = "validator"
    return validator


def create_tester(model: torch.nn.Module, config: dict) -> Engine:
    """This function creates a Pytorch Ignite engine object that will be used to
    test the model and compute metrics without updating model weights.

    Parameters
    ----------
    model : torch.nn.Module
        The model to test
    config : dict
        Hyrax runtime configuration

    Returns
    -------
    pytorch-ignite.Engine
        Engine object that will be used to test the model and compute metrics.
    """

    device = idist.device()
    wrapped_model = idist.auto_model(model)
    tensorboardx_logger = get_tensorboard_logger()

    tester = create_engine("test_batch", device, wrapped_model, config)
    fixup_engine(tester)

    @tester.on(Events.STARTED)
    def set_model_to_eval_mode():
        wrapped_model.eval()

    # Track average loss
    from ignite.metrics import RunningAverage

    RunningAverage(output_transform=lambda x: x["loss"]).attach(tester, "avg_loss")

    @tester.on(Events.STARTED)
    def log_test_start(engine):
        logger.info(f"Starting model evaluation on test data (device: {device})")

    # Wrap iteration to disable gradients during testing
    original_run = tester.run

    def run_with_no_grad(data, *args, **kwargs):
        with torch.no_grad():
            return original_run(data, *args, **kwargs)

    tester.run = run_with_no_grad

    @tester.on(Events.COMPLETED)
    def log_test_metrics(engine):
        from colorama import Fore, Style

        metrics = engine.state.metrics
        logger.info(f"{Style.BRIGHT}{Fore.GREEN}Test Results:{Style.RESET_ALL}")
        logger.info(f"  Average Loss: {metrics.get('avg_loss', 'N/A'):.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("avg_loss", metrics.get("avg_loss", 0.0))

        # Log to tensorboard
        tensorboardx_logger.add_scalar("test/avg_loss", metrics.get("avg_loss", 0.0), 0)

    tester.hyrax_label = "tester"
    return tester


def attach_best_checkpoint(
    engine: Engine,
    model: torch.nn.Module,
    trainer: Engine,
    results_directory: Path,
) -> None:
    """Attach a best-checkpoint handler to ``engine``, scored on ``engine.state.output["loss"]``.

    Call this function *after* both ``create_trainer`` and (optionally) ``create_validator``
    have been called so that handler registration order is correct.  When a validator is
    available, pass it as ``engine`` so that checkpointing is driven by validation loss.
    When no validator is available, pass the trainer as ``engine`` so that checkpointing
    falls back to training loss — preserving the previous behaviour.

    The saved checkpoint format is identical to the one produced by ``create_trainer``, so
    existing resume logic is fully backward-compatible.

    Parameters
    ----------
    engine : pytorch-ignite.Engine
        The engine whose ``output["loss"]`` is used as the checkpoint score.  Pass the
        validator when one exists; otherwise pass the trainer. If the engine has a
        ``hyrax_label`` attribute, it will be included in the checkpoint filename.
    model : torch.nn.Module
        The model being trained.  Must expose ``model.optimizer`` and optionally
        ``model.scheduler``.
    trainer : pytorch-ignite.Engine
        The training engine.  Used to derive the global step counter and to attach the
        end-of-training log handler.
    results_directory : Path
        Directory where checkpoint files are written.
    """
    wrapped_model = idist.auto_model(model)

    to_save = {
        "model": wrapped_model,
        "optimizer": model.optimizer,
        "trainer": trainer,
    }

    if model.scheduler:
        to_save["scheduler"] = model.scheduler

    def neg_loss_score(eng):
        return -eng.state.output["loss"]

    score_name = f"{engine.hyrax_label}_loss" if hasattr(engine, "hyrax_label") else "loss"

    best_checkpoint = Checkpoint(
        to_save,
        DiskSaver(results_directory, require_empty=False),
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
        score_name=score_name,
        score_function=neg_loss_score,
        greater_or_equal=True,
    )

    engine.add_event_handler(HyraxEvents.HYRAX_EPOCH_COMPLETED, best_checkpoint)

    def log_best_checkpoint_location(_, chkpt):
        logger.debug(f"Best metric checkpoint saved as: {chkpt.last_checkpoint}")

    trainer.add_event_handler(Events.COMPLETED, log_best_checkpoint_location, best_checkpoint)


def create_trainer(model: torch.nn.Module, config: dict, results_directory: Path) -> Engine:
    """This function is originally copied from here:
    https://github.com/pytorch-ignite/examples/blob/main/tutorials/intermediate/cifar10-distributed.py#L164

    It was substantially trimmed down to make it easier to understand.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    config : dict
        Hyrax runtime configuration
    results_directory : Path
        The directory where training results will be saved

    Returns
    -------
    pytorch-ignite.Engine
        Engine object that will be used to train the model.
    """
    device = idist.device()
    model.train()
    wrapped_model = idist.auto_model(model)
    trainer = create_engine("train_batch", device, wrapped_model, config)
    tensorboardx_logger = get_tensorboard_logger()
    fixup_engine(trainer)

    to_save = {
        "model": wrapped_model,
        "optimizer": model.optimizer,
        "trainer": trainer,
    }

    if model.scheduler:
        to_save["scheduler"] = model.scheduler

    latest_checkpoint = Checkpoint(
        to_save,
        DiskSaver(results_directory, require_empty=False),
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
        filename_pattern="{name}_epoch_{global_step}.{ext}",
    )

    if config["train"]["resume"]:
        # Load checkpoint with weights_only=False because pytorch-ignite checkpoints
        # contain optimizer and trainer state objects, not just model weights.
        # This is different from loading just model weights, which would use weights_only=True.
        prev_checkpoint = torch.load(config["train"]["resume"], map_location=device, weights_only=False)
        Checkpoint.load_objects(to_load=to_save, checkpoint=prev_checkpoint)

    @trainer.on(Events.STARTED)
    def log_training_start(trainer):
        logger.debug(f"Training model on device: {device}")

    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch_start(trainer):
        logger.debug(f"Starting epoch {trainer.state.epoch}")

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss_tensorboard(trainer):
        step = trainer.state.get_event_attrib_value(Events.ITERATION_COMPLETED)
        for m in trainer.state.output:
            tensorboardx_logger.add_scalar(f"training/training/{m}", trainer.state.output[m], step)
            mlflow.log_metrics({f"training/{m}": trainer.state.output[m]}, step=step)

    @trainer.on(HyraxEvents.HYRAX_EPOCH_COMPLETED)
    def log_training_loss(trainer):
        logger.debug(f"Epoch {trainer.state.epoch} run time: {trainer.state.times['EPOCH_COMPLETED']:.2f}[s]")
        logger.debug(f"Epoch {trainer.state.epoch} metrics: {trainer.state.output}")

    @trainer.on(HyraxEvents.HYRAX_EPOCH_COMPLETED)
    def log_epoch_metrics(trainer):
        if hasattr(model, "log_epoch_metrics"):
            epoch_number = trainer.state.epoch
            epoch_metrics = model.log_epoch_metrics()
            for m in epoch_metrics:
                tensorboardx_logger.add_scalar(
                    f"training/training/epoch/{m}", epoch_metrics[m], global_step=epoch_number
                )
                mlflow.log_metrics({f"training/epoch/{m}": epoch_metrics[m]}, step=epoch_number)

    @trainer.on(HyraxEvents.HYRAX_EPOCH_COMPLETED)
    def scheduler_step(trainer):
        if model.scheduler:
            if not hasattr(model, "_learning_rates_history"):
                model._learning_rates_history = []
            epoch_lr = model.scheduler.get_last_lr()
            epoch_number = trainer.state.epoch - 1
            model._learning_rates_history.append(epoch_lr)
            tensorboardx_logger.add_scalar("training/training/epoch/lr", epoch_lr, global_step=epoch_number)
            model.scheduler.step()

    trainer.add_event_handler(HyraxEvents.HYRAX_EPOCH_COMPLETED, latest_checkpoint)

    @trainer.on(Events.COMPLETED)
    def log_total_time(trainer):
        logger.info(f"Total training time: {trainer.state.times['COMPLETED']:.2f}[s]")

    def log_last_checkpoint_location(_, latest_checkpoint):
        logger.debug(f"Latest checkpoint saved as: {latest_checkpoint.last_checkpoint}")

    trainer.add_event_handler(Events.COMPLETED, log_last_checkpoint_location, latest_checkpoint)

    @trainer.on(Events.COMPLETED)
    def attach_final_metrics_to_model(trainer):
        # Attach the final training metrics to the model object for easy access
        model.final_training_metrics = trainer.state.output

    pbar = ProgressBar(persist=False, bar_format="")
    pbar.attach(trainer)

    trainer.hyrax_label = "trainer"
    return trainer


def create_save_batch_callback(results_dir):
    """Create a callback function for saving batch results during inference or testing.

    This factory function creates a closure that captures the output directory,
    then returns a callback that can be used by pytorch_ignite engines to save
    model outputs batch by batch.

    Parameters
    ----------
    results_dir : Path
        Directory where results should be saved

    Returns
    -------
    callable
        A callback function with signature (batch, batch_results) that saves results
    """
    from hyrax.datasets.result_factories import create_results_writer

    data_writer = create_results_writer(results_dir)

    def _save_batch(batch: dict, batch_results: torch.Tensor):
        """Receive and write batch results to results_dir immediately."""
        nonlocal data_writer

        # Ensure the batch results are on CPU and detached from the computation graph
        batch_results = batch_results.detach().to("cpu")

        # Verify that batch contains object_id
        if "object_id" not in batch:
            msg = "The data batch is missing the key: 'object_id'. "
            msg += "Cannot save the model output."
            logger.error(msg)
            raise RuntimeError(msg)

        batch_object_ids = batch["object_id"]

        # Ensure that everything to be written is in numpy format, and write it out
        data_writer.write_batch(np.array(batch_object_ids), [t.numpy() for t in batch_results])

    # Attach the data_writer to the callback so it can be accessed later
    _save_batch.data_writer = data_writer  # type: ignore[attr-defined]

    return _save_batch


class HyraxEvents(EventEnum):
    """
    Workaround event for a pytorch ignite bug. See fixup_engine for details
    """

    HYRAX_EPOCH_COMPLETED = "HyraxEpochCompleted"


def fixup_engine(engine: Engine):
    """
    Workaround for this pytorch ignite bug (https://github.com/pytorch/ignite/issues/3372) where
    engine.state.output is not available at EPOCH_COMPLETED or later times (COMPLETED, etc)

    We create a new event HYRAX_EPOCH_COMPLETED which triggers at ITERATION_COMPLETED, but only on the final
    iteration. This is just before the erronious state reset.

    This hack relies on pytorch ignite internal state, but can be removed as soon as our fix is mainlined
    (https://github.com/pytorch/ignite/pull/3373) in version 0.6.0 estimated August 2025
    """
    from more_itertools import peekable

    engine.register_events(*HyraxEvents)

    @engine.on(Events.ITERATION_COMPLETED)
    def maintain_event_handler(engine):
        # Ensure we have a peekable iterator in the engine.
        if not hasattr(engine._dataloader_iter, "peek"):
            # Replace with a pass-through peekable iterator
            engine._dataloader_iter = peekable(engine._dataloader_iter)

        # On the last iteration the peekable iterator evaluates as true
        if not engine._dataloader_iter:
            engine.fire_event(HyraxEvents.HYRAX_EPOCH_COMPLETED)
