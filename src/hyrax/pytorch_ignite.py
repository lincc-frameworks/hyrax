import functools
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Union

import ignite.distributed as idist
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    import mlflow

import torch
from ignite.engine import Engine, EventEnum, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.tqdm_logger import ProgressBar
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from hyrax.config_utils import ConfigDict
from hyrax.data_sets.data_set_registry import HyraxDataset, fetch_data_set_class
from hyrax.models.model_registry import fetch_model_class

logger = logging.getLogger(__name__)


def setup_dataset(config: ConfigDict, tensorboardx_logger: Optional[SummaryWriter] = None) -> Dataset:
    """Create a dataset object based on the configuration.

    Parameters
    ----------
    config : ConfigDict
        The entire runtime configuration
    tensorboardx_logger : SummaryWriter, optional
        If Tensorboard is in use, the tensorboard logger so the dataset can log things

    Returns
    -------
    Dataset
        An instance of the dataset class specified in the configuration
    """

    # Fetch data loader class specified in config and create an instance of it
    data_set_cls = fetch_data_set_class(config)
    data_set: HyraxDataset = data_set_cls(config)  # type: ignore[call-arg]

    data_set.tensorboardx_logger = tensorboardx_logger

    return data_set


def setup_model(config: ConfigDict, dataset: Dataset) -> torch.nn.Module:
    """Create a model object based on the configuration.

    Parameters
    ----------
    config : ConfigDict
        The entire runtime configuration
    dataset : Dataset
        Only used to determine the input shape of the data

    Returns
    -------
    torch.nn.Module
        An instance of the model class specified in the configuration
    """

    # Fetch model class specified in config and create an instance of it
    model_cls = fetch_model_class(config)
    model = model_cls(config=config, shape=dataset.shape())  # type: ignore[attr-defined]

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
        return idist.auto_dataloader(data_set, sampler=None, **config["data_loader"])

    # Sanitize split argument
    if isinstance(split, str):
        split = [split]

    # Configure the torch rng
    torch_rng = torch.Generator()
    seed = config["data_set"]["seed"] if config["data_set"]["seed"] else None
    if seed is not None:
        torch_rng.manual_seed(seed)

    if data_set.is_iterable():
        dataloaders = {
            s: idist.auto_dataloader(data_set, pin_memory=True, **config["data_loader"]) for s in split
        }
    else:
        # Create the indexes for all splits based on config.
        indexes = create_splits(data_set, config)

        # Create samplers and dataloaders for each split we are interested in
        samplers = {
            s: SubsetRandomSampler(indexes[s], generator=torch_rng) if indexes.get(s) else None for s in split
        }

        dataloaders = {
            split: idist.auto_dataloader(data_set, sampler=sampler, **config["data_loader"])
            if sampler
            else None
            for split, sampler in samplers.items()
        }

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


def create_engine(funcname: str, device: torch.device, model: torch.nn.Module) -> Engine:
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
    """

    # This wraps a model-specific function (func) to move data to the appropriate device.
    def _inner_loop(func, device, engine, batch):
        #! This feels brittle, it would be worth revisiting this.
        #  We assume that the batch data will generally have two forms.
        # 1) A torch.Tensor that represents N samples.
        # 2) A tuple (or list) of torch.Tensors, where the first tensor is the
        # data, and the second is labels.
        batch = batch.to(device) if isinstance(batch, torch.Tensor) else tuple(i.to(device) for i in batch)
        return func(batch)

    def _create_process_func(funcname, device, model):
        inner_step = extract_model_method(model, funcname)
        inner_loop = functools.partial(_inner_loop, inner_step, device)
        return inner_loop

    return Engine(_create_process_func(funcname, device, model))


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


def create_evaluator(model: torch.nn.Module, save_function: Callable[[torch.Tensor], Any]) -> Engine:
    """Creates an evaluator engine
    Primary purpose of this function is to attach the appropriate handlers to an evaluator engine

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate

    save_function : Callable[[torch.Tensor], Any]
        A function which will receive Engine.state.output at the end of each iteration. The intent
        is for the results of evaluation to be saved.

    Returns
    -------
    pytorch-ignite.Engine
        Engine object which when run will evaluate the model.
    """
    device = idist.device()
    model.eval()
    model = idist.auto_model(model)
    evaluator = create_engine("forward", device, model)

    @evaluator.on(Events.STARTED)
    def log_eval_start(evaluator):
        logger.info(f"Evaluating model on device: {device}")
        logger.info(f"Total epochs: {evaluator.state.max_epochs}")

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_iteration_complete(evaluator):
        save_function(evaluator.state.output)

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

    validator = create_engine("train_step", device, model)
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
        tensorboardx_logger.add_scalar("training/validation/loss", validator.state.output["loss"], step)
        mlflow.log_metrics({"validation/loss": validator.state.output["loss"]}, step=step)

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
    trainer = create_engine("train_step", device, model)
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
        print(engine.state)
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

    # results_root_dir = Path(config["general"]["results_dir"]).expanduser().resolve()
    # mlflow_logger = MLflowLogger("file://" + str(results_root_dir / "mlflow"))

    @trainer.on(Events.STARTED)
    def log_training_start(trainer):
        logger.info(f"Training model on device: {device}")

    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch_start(trainer):
        logger.debug(f"Starting epoch {trainer.state.epoch}")

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss_tensorboard(trainer):
        step = trainer.state.get_event_attrib_value(Events.ITERATION_COMPLETED)
        tensorboardx_logger.add_scalar("training/training/loss", trainer.state.output["loss"], step)
        mlflow.log_metrics({"training/loss": trainer.state.output["loss"]}, step=step)

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
        logger.info(f"Latest checkpoint saved as: {latest_checkpoint.last_checkpoint}")

    def log_best_checkpoint_location(_, best_checkpoint):
        logger.info(f"Best metric checkpoint saved as: {best_checkpoint.last_checkpoint}")

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
