import functools
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Union

import ignite.distributed as idist
import numpy as np

from hyrax.data_sets.data_set_registry import fetch_dataset_class

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    import mlflow

from collections.abc import Iterator, Sequence

import torch
from ignite.engine import Engine, EventEnum, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.tqdm_logger import ProgressBar
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from hyrax.config_utils import ConfigDict
from hyrax.data_sets.data_provider import DataProvider, generate_data_request_from_config
from hyrax.models.model_registry import fetch_model_class

logger = logging.getLogger(__name__)


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


def is_iterable_dataset_requested(data_request: dict) -> bool:
    """This function checks each of the datasets included in the data_request.
    If any of them are iterable-style datasets, we return True.
    """

    is_iterable = False
    for _, dataset_definition in data_request.items():
        if fetch_dataset_class(dataset_definition["dataset_class"]).is_iterable():
            is_iterable = True
    return is_iterable


def setup_dataset(config: ConfigDict, tensorboardx_logger: Optional[SummaryWriter] = None) -> Dataset:
    """This function creates an instance of the requested dataset specified in the
    runtime configuration. There are two modes encapsulated here:

    1) If the dataset requested includes an iterable-style dataset, ensure that only
    one dataset was requested, and then return an instance of that dataset.
    2) If the dataset(s) requested is for 1 or more map-style dataset, create an
    instance of a DataProvider, and return that as the dataset.

    Parameters
    ----------
    config : ConfigDict
        The runtime configuration
    tensorboardx_logger : SummaryWriter, optional
        If Tensorboard is in use, the tensorboard logger so the dataset can log things

    Returns
    -------
    Dataset
        An instance of the dataset class specified in the configuration
    """

    data_request = generate_data_request_from_config(config)
    if is_iterable_dataset_requested(data_request):
        # If the data_request is for multiple datasets and at least one of
        # them is iterable, raise an error, we don't support that style of operation
        if len(data_request) > 1:
            logger.error(
                "Multiple datasets requested, including at least one iterable-style. "
                "Hyrax supports for datasets includes: "
                "1) 1-N map-style or 2) at most 1 iterable-style."
            )
            raise RuntimeError(
                "Multiple datasets requested, including at least one iterable-style. "
                "Hyrax supports for datasets includes: "
                "1) 1-N map-style or 2) at most 1 iterable-style."
            )

        # generate instance of the iterable dataset. Again, because the only mode of
        # operation for iterable-style datasets that Hyrax supports is 1 iterable
        # dataset at a time, we can just take the first (and only) item in the data_request.
        data_definition = next(iter(data_request.values()))

        dataset_class = data_definition.get("dataset_class", None)

        dataset_cls = fetch_dataset_class(dataset_class)

        data_location = data_definition.get("data_location", None)

        dataset = dataset_cls(config=config, data_location=data_location)
        dataset.tensorboardx_logger = tensorboardx_logger

    else:
        dataset = DataProvider(config)
        for friendly_name in dataset.prepped_datasets:
            dataset.prepped_datasets[friendly_name].tensorboardx_logger = tensorboardx_logger

    return dataset


def setup_model(config: ConfigDict, dataset: Dataset) -> torch.nn.Module:
    """Create a model object based on the configuration.

    Parameters
    ----------
    config : ConfigDict
        The runtime configuration
    dataset : Dataset
        The dataset object that will provide data to the model for training or
        inference. Here it is only used to provide a data sample to the model so
        that it can resize itself at runtime if necessary.

    Returns
    -------
    torch.nn.Module
        An instance of the model class specified in the configuration
    """

    # Fetch model class specified in config and create an instance of it
    model_cls = fetch_model_class(config)
    model = model_cls(config=config, data_sample=dataset.sample_data())  # type: ignore[attr-defined]

    return model


def dist_data_loader(
    data_set: Dataset,
    config: ConfigDict,
    split: Union[str, list[str], bool] = False,
):
    """Create Pytorch Ignite distributed data loaders

    It is recommended that each verb needing dataloaders only call this function once.

    Parameters
    ----------
    data_set : Dataset
        A Pytorch Dataset object
    config : ConfigDict
        Hyrax runtime configuration
    split : Union[str, list[str]], Optional
        The name(s) of the split we want to use from the data set.
        If this is false or not passed, then a single data loader is returned
        that corresponds to the entire dataset.

    Returns
    -------
    Dataloader (or an ignite-wrapped equivalent)
        This is the distributed dataloader, formed by calling ignite.distributed.auto_dataloader

    For multiple splits, we return a dictionary where the keys are the names of the splits
    and the value is either a Dataloader as described above or the value None if the split
    was not configured.

    If an iterable dataset is passed, we cannot create multiple splits with a pyTorch sampler object
    so we return the same thing for all splits, which is a dataloader representing the entire iterable
    """
    # Handle case where no split is needed.
    if isinstance(split, bool):
        # We still need to return the list of indexes used by the dataloader,
        # but here, it will simply be the indexes for the entire dataset.
        if data_set.is_iterable():
            ids = list(data_set.ids())
            indexes = list(range(len(ids)))
        else:
            indexes = list(range(len(data_set)))

        # Note that when sampler=None, a default sampler is used. The default config
        # defines shuffle=False, which should prevent any shuffling of of the data.
        # We expect that this will be the primary use case when running inference.
        return idist.auto_dataloader(data_set, sampler=None, **config["data_loader"]), indexes

    # Sanitize split argument
    if isinstance(split, str):
        split = [split]

    # Configure the torch rng
    torch_rng = torch.Generator()
    seed = config["data_set"]["seed"] if config["data_set"]["seed"] else None
    if seed is not None:
        torch_rng.manual_seed(seed)

    if data_set.is_iterable():
        ids = list(data_set.ids())
        indexes = list(range(len(ids)))
        dataloaders = {
            s: (idist.auto_dataloader(data_set, pin_memory=True, **config["data_loader"]), indexes)
            for s in split
        }
    else:
        # Create the indexes for all splits based on config.
        indexes = create_splits(data_set, config)

        # Create samplers and dataloaders for each split we are interested in
        samplers = {s: SubsetSequentialSampler(indexes[s]) if indexes.get(s) else None for s in split}

        dataloaders = {
            split: (idist.auto_dataloader(data_set, sampler=sampler, **config["data_loader"]), indexes[split])
            if sampler
            else None
            for split, sampler in samplers.items()
        }

        none_keys = [k for k, v in dataloaders.items() if v is None]
        for key in none_keys:
            del dataloaders[key]

    # Return only one if we were only passed one split in, return the dictionary otherwise.
    return dataloaders[split[0]] if len(split) == 1 else dataloaders


def create_splits(data_set: Dataset, config: ConfigDict):
    """Returns train, test, and validation indexes constructed to be used with the passed in
    dataset. The allocation of indexes in the underlying dataset to samplers depends on
    the data_set section of the config dict.

    Parameters
    ----------
    data_set : Dataset
        The data set to use
    config : ConfigDict
        Configuration that defines dataset splits
    split : str
        Name of the split to use.
    """
    data_set_size = len(data_set)  # type: ignore[arg-type]

    # Init the splits based on config values
    train_size = config["data_set"]["train_size"] if config["data_set"]["train_size"] else None
    test_size = config["data_set"]["test_size"] if config["data_set"]["test_size"] else None
    validate_size = config["data_set"]["validate_size"] if config["data_set"]["validate_size"] else None

    # Convert all values specified as counts into ratios of the underlying container
    if isinstance(train_size, int):
        train_size = train_size / data_set_size
    if isinstance(test_size, int):
        test_size = test_size / data_set_size
    if isinstance(validate_size, int):
        validate_size = validate_size / data_set_size

    # Initialize Test size when not provided
    if test_size is None:
        if train_size is None:
            train_size = 0.25

        if validate_size is None:  # noqa: SIM108
            test_size = 1.0 - train_size
        else:
            test_size = 1.0 - (train_size + validate_size)

    # Initialize train size when not provided, and can be inferred from test_size and validate_size.
    if train_size is None:
        if validate_size is None:  # noqa: SIM108
            train_size = 1.0 - test_size
        else:
            train_size = 1.0 - (test_size + validate_size)

    # If splits cover more than the entire dataset, error out.
    if validate_size is None:
        if np.round(train_size + test_size, decimals=5) > 1.0:
            raise RuntimeError("Split fractions add up to more than 1.0")
    elif np.round(train_size + test_size + validate_size, decimals=5) > 1.0:
        raise RuntimeError("Split fractions add up to more than 1.0")

    # If any split is less than 0.0 also error out
    if (
        np.round(test_size, decimals=5) < 0.0
        or np.round(train_size, decimals=5) < 0.0
        or (validate_size is not None and np.round(validate_size, decimals=5) < 0.0)
    ):
        raise RuntimeError("One of the Split fractions configured is negative.")

    indices = list(range(data_set_size))

    # shuffle the indices
    seed = config["data_set"]["seed"] if config["data_set"]["seed"] else None
    np.random.seed(seed)
    np.random.shuffle(indices)

    # Given the number of samples in the dataset and the ratios of the splits
    # we can calculate the number of samples in each split.
    num_test = int(np.round(data_set_size * test_size))
    num_train = int(np.round(data_set_size * train_size))

    # split the indices
    test_idx = indices[:num_test]
    train_idx = indices[num_test : num_test + num_train]

    # assume that validate gets all the remaining indices
    if validate_size:
        num_validate = int(np.round(data_set_size * validate_size))
        valid_idx = indices[num_test + num_train : num_test + num_train + num_validate]

    split_inds = {"train": train_idx, "test": test_idx}
    if validate_size:
        split_inds["validate"] = valid_idx

    return split_inds


@functools.singledispatch
def _handle_nans(batch, config):
    """The default _handle_nan function. Will print a warning and return `batch`."""
    logger.warning(
        f"Encountered an unhandled batch type, {type(batch)}, while\
                   attempting to handle NaN values in the data."
    )
    return batch


@_handle_nans.register(torch.Tensor)
def _handle_nans_tensor(batch, config):
    """The implementation of _handle_nans when expecting `batch` to be a tensor."""
    return _handle_nans_logic(batch, config)


@_handle_nans.register(tuple)
def _handle_nans_tuple(batch, config):
    """This is the tuple-specific implementation of _handle_nans. The first element
    of the tuple is expected to hold the data that needs nan-cleaning."""
    nan_handled = _handle_nans_logic(batch[0], config)
    return (nan_handled, batch[1])


def _handle_nans_logic(batch, config):
    from torch import any, isnan

    if config["data_set"]["nan_mode"] is False:
        if any(isnan(batch)):
            msg = "Input data contains NaN values. This may mean your model output is all NaNs."
            msg += "Consider setting config['data_set']['nan_mode'] = 'quantile' or 'zero' or writing a "
            msg += "to_tensor() function for your model. Search hyrax readthedocs for 'to_tensor' "
            msg += "to get started."
            logger.warning(msg)
        return batch

    if config["data_set"]["nan_mode"] == "quantile":
        quantile = config["data_set"]["nan_quantile"]
        if quantile < 0.0 or quantile > 1.0:
            raise RuntimeError('set config["data_set"]["nan_quantile"] to a value between 0 and 1')
        return _handle_nan_quantile(batch, quantile)
    elif config["data_set"]["nan_mode"] == "zero":
        return _handle_nan_zero(batch)
    else:
        msg = f"nan mode was set to '{config['data_set']['nan_mode']}' which is unsupported."
        msg += "The supported modes are 'quantile' and 'zero'."
        raise NotImplementedError(msg)


def _handle_nan_quantile(batch, quantile):
    from torch import any, isnan

    if any(isnan(batch)):
        flat_batch = torch.reshape(batch, (batch.shape[0], -1))
        batch_quantile = torch.nanquantile(flat_batch, q=quantile, dim=-1)
        for i, val in enumerate(batch_quantile):
            batch[i] = torch.nan_to_num(batch[i], val)

    return batch


def _handle_nan_zero(batch):
    from torch import any, isnan

    if any(isnan(batch)):
        batch = torch.nan_to_num(batch, nan=0.0)

    return batch


def _inner_loop(func, to_tensor, device, config, engine, batch):
    """This wraps a model-specific function (func) to move data to the appropriate device."""
    # If we have a dict of lists, we need to decode the dict
    # This will give us either a tensor of the whole batch or a tuple
    if isinstance(batch, dict):
        batch = to_tensor(batch)

    batch = _handle_nans(batch, config)

    # Send the batch to the device
    batch = batch.to(device) if isinstance(batch, torch.Tensor) else tuple(i.to(device) for i in batch)
    return func(batch)


def _create_process_func(funcname, device, model, config):
    inner_step = extract_model_method(model, funcname)
    to_tensor = extract_model_method(model, "to_tensor")
    inner_loop = functools.partial(_inner_loop, inner_step, to_tensor, device, config)
    return inner_loop


def create_engine(funcname: str, device: torch.device, model: torch.nn.Module, config: ConfigDict) -> Engine:
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
    config : ConfigDict
        The runtime config in use
    """
    return Engine(_create_process_func(funcname, device, model, config))


def extract_model_method(model, method_name):
    """Extract a method from a model, which may be wrapped in a DistributedDataParallel
    or DataParallel object. For instance, method_name could be `train_step` or
    `forward`.

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
    return getattr(model.module if wrapped else model, method_name)


def create_evaluator(
    model: torch.nn.Module, save_function: Callable[[torch.Tensor, torch.Tensor], Any], config: ConfigDict
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

    config : ConfigDict
        The runtime config in use

    Returns
    -------
    pytorch-ignite.Engine
        Engine object which when run will evaluate the model.
    """
    device = idist.device()
    model.eval()
    model = idist.auto_model(model)
    evaluator = create_engine("forward", device, model, config)

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

    return evaluator


#! There will likely be a significant amount of code duplication between the
#! `create_trainer` and `create_validator` functions. We should find a way to
#! refactor this code to reduce duplication.
def create_validator(
    model: torch.nn.Module,
    config: ConfigDict,
    results_directory: Path,
    tensorboardx_logger: SummaryWriter,
    validation_data_loader: DataLoader,
    trainer: Engine,
) -> Engine:
    """This function creates a Pytorch Ignite engine object that will be used to
    validate the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    config : ConfigDict
        Hyrax runtime configuration
    results_directory : Path
        The directory where training results will be saved
    tensorboardx_logger : SummaryWriter
        The tensorboard logger object
    validation_data_loader : DataLoader
        The data loader for the validation data
    trainer : Engine
        The engine object that will be used to train the model. We will use specific
        hooks in the trainer to determine when to run the validation engine.

    Returns
    -------
    pytorch-ignite.Engine
        Engine object that will be used to train the model.
    """

    device = idist.device()
    model = idist.auto_model(model)

    validator = create_engine("train_step", device, model, config)
    fixup_engine(validator)

    @validator.on(Events.STARTED)
    def set_model_to_eval_mode():
        model.eval()

    @validator.on(Events.COMPLETED)
    def set_model_to_train_mode():
        model.train()

    @validator.on(HyraxEvents.HYRAX_EPOCH_COMPLETED)
    def log_training_loss():
        logger.debug(f"Validation run time: {validator.state.times['EPOCH_COMPLETED']:.2f}[s]")
        logger.debug(f"Validation metrics: {validator.state.output}")

    @trainer.on(HyraxEvents.HYRAX_EPOCH_COMPLETED)
    def run_validation():
        validator.run(validation_data_loader)

    def log_validation_loss(validator, trainer):
        step = trainer.state.get_event_attrib_value(Events.EPOCH_COMPLETED)
        for m in trainer.state.output:
            tensorboardx_logger.add_scalar(f"training/validation/{m}", validator.state.output[m], step)
            mlflow.log_metrics({f"validation/{m}": validator.state.output[m]}, step=step)

    validator.add_event_handler(HyraxEvents.HYRAX_EPOCH_COMPLETED, log_validation_loss, trainer)

    return validator


def create_trainer(
    model: torch.nn.Module, config: ConfigDict, results_directory: Path, tensorboardx_logger: SummaryWriter
) -> Engine:
    """This function is originally copied from here:
    https://github.com/pytorch-ignite/examples/blob/main/tutorials/intermediate/cifar10-distributed.py#L164

    It was substantially trimmed down to make it easier to understand.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    config : ConfigDict
        Hyrax runtime configuration
    results_directory : Path
        The directory where training results will be saved
    tensorboardx_logger : SummaryWriter
        The tensorboard logger object

    Returns
    -------
    pytorch-ignite.Engine
        Engine object that will be used to train the model.
    """
    device = idist.device()
    model.train()
    model = idist.auto_model(model)
    trainer = create_engine("train_step", device, model, config)
    fixup_engine(trainer)

    optimizer = extract_model_method(model, "optimizer")

    to_save = {
        "model": model,
        "optimizer": optimizer,
        "trainer": trainer,
    }

    #! We may want to move the checkpointing logic over to the `validator`.
    #! It was created here initially because this was the only place where the
    #! model training was happening.
    latest_checkpoint = Checkpoint(
        to_save,
        DiskSaver(results_directory, require_empty=False),
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
        filename_pattern="{name}_epoch_{global_step}.{ext}",
    )

    def neg_loss_score(engine):
        return -engine.state.output["loss"]

    best_checkpoint = Checkpoint(
        to_save,
        DiskSaver(results_directory, require_empty=False),
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
        score_name="loss",
        score_function=neg_loss_score,
        greater_or_equal=True,
    )

    if config["train"]["resume"]:
        prev_checkpoint = torch.load(config["train"]["resume"], map_location=device)
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

    trainer.add_event_handler(HyraxEvents.HYRAX_EPOCH_COMPLETED, latest_checkpoint)
    trainer.add_event_handler(HyraxEvents.HYRAX_EPOCH_COMPLETED, best_checkpoint)

    @trainer.on(Events.COMPLETED)
    def log_total_time(trainer):
        logger.info(f"Total training time: {trainer.state.times['COMPLETED']:.2f}[s]")

    def log_last_checkpoint_location(_, latest_checkpoint):
        logger.debug(f"Latest checkpoint saved as: {latest_checkpoint.last_checkpoint}")

    def log_best_checkpoint_location(_, best_checkpoint):
        logger.debug(f"Best metric checkpoint saved as: {best_checkpoint.last_checkpoint}")

    trainer.add_event_handler(Events.COMPLETED, log_last_checkpoint_location, latest_checkpoint)
    trainer.add_event_handler(Events.COMPLETED, log_best_checkpoint_location, best_checkpoint)

    pbar = ProgressBar(persist=False, bar_format="")
    pbar.attach(trainer)

    return trainer


class HyraxEvents(EventEnum):
    """
    Workaround event for a pytorch ignite bug. See fixup_engine for details
    """

    HYRAX_EPOCH_COMPLETED = "HyraxEpochCompleted"


def fixup_engine(engine: Engine) -> Engine:
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
