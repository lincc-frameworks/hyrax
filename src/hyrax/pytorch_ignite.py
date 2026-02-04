import functools
import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

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
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from hyrax.data_sets.data_provider import DataProvider, generate_data_request_from_config
from hyrax.models.model_registry import fetch_model_class
from hyrax.plugin_utils import get_or_load_class
from hyrax.tensorboardx_logger import get_tensorboard_logger

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
    for _, value in data_request.items():
        for _, dataset_definition in value.items():
            if fetch_dataset_class(dataset_definition["dataset_class"]).is_iterable():
                is_iterable = True
                break
    return is_iterable


def setup_dataset(config: dict) -> Dataset:
    """This function creates an instance of the requested dataset specified in the
    runtime configuration. There are two modes encapsulated here:

    1) If the dataset requested includes an iterable-style dataset, ensure that only
    one dataset was requested, and then return an instance of that dataset.
    2) If the dataset(s) requested is for 1 or more map-style dataset, create an
    instance of a DataProvider, and return that as the dataset.

    Parameters
    ----------
    config : dict
        The runtime configuration

    Returns
    -------
    Dataset
        An instance of the dataset class specified in the configuration
    """

    dataset = {}
    data_request = generate_data_request_from_config(config)
    if is_iterable_dataset_requested(data_request):
        # If the data_request is for multiple datasets and at least one of
        # them is iterable, raise an error, we don't support that style of operation
        for _, value in data_request.items():
            if len(value) > 1:
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
        for set_name in ["train", "infer"]:
            data_definition = next(iter(data_request[set_name].values()))

            dataset_class = data_definition.get("dataset_class", None)
            dataset_cls = fetch_dataset_class(dataset_class)

            data_location = data_definition.get("data_location", None)
            ds = dataset_cls(config=config, data_location=data_location)

            dataset[set_name] = ds

    else:
        # We know that `data_request` will always have at least 2 sub-tables, `train`
        # and `infer`. It may have additional sub-tables such as `validate`.
        for key, value in data_request.items():
            ds = DataProvider(config, value)
            dataset[key] = ds

    return dataset


def setup_model(config: dict, dataset: Dataset) -> torch.nn.Module:
    """Create a model object based on the configuration.

    Parameters
    ----------
    config : dict
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

    # Pass a single sample of data through the model's prepare_inputs function
    # ? I don't think that the `if` portion of this logic is used, should double check
    if isinstance(dataset, dict):
        # If we have multiple datasets, just take the first one
        first_dataset = next(iter(dataset.values()))
        data_sample = model_cls.prepare_inputs(first_dataset.sample_data())
    else:
        data_sample = model_cls.prepare_inputs(dataset.sample_data())

    # Provide the data sample for runtime modifications to the model architecture
    return model_cls(config=config, data_sample=data_sample)  # type: ignore[attr-defined]


def load_collate_function(data_loader_kwargs: dict) -> Callable | None:
    """Load a collate function if one is specified in the config. Otherwise return None.
    Returning None will cause the DataLoader to use PyTorch's default collate function.

    Parameters
    ----------
    data_loader_kwargs : dict
        The configuration dictionary that will be passed as kwargs to the DataLoader

    Returns
    -------
    Optional[Callable]
        The collate function if specified, else None
    """
    collate_fn = (
        get_or_load_class(data_loader_kwargs["collate_fn"]) if data_loader_kwargs["collate_fn"] else None
    )
    return collate_fn


def dist_data_loader(
    dataset: Dataset,
    config: dict,
    split: Union[str, list[str], bool] = False,
):
    """Create Pytorch Ignite distributed data loaders

    It is recommended that each verb needing dataloaders only call this function once.

    Parameters
    ----------
    dataset : hyrax.data_sets.data_set_registry.HyraxDataset
        A Hyrax dataset instance
    config : dict
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

    # Extract the config dictionary that will be provided as kwargs to the DataLoader
    data_loader_kwargs = dict(config["data_loader"])

    # If the dataset is a DataProvider instance, use its collate function.
    # Else use the collate function defined in the config, or None (Torch's default)
    if isinstance(dataset, DataProvider):
        collation_func = dataset.collate
    else:
        collation_func = load_collate_function(data_loader_kwargs)
    data_loader_kwargs["collate_fn"] = collation_func

    # Handle case where no split is needed.
    if isinstance(split, bool):
        # We still need to return the list of indexes used by the dataloader,
        # but here, it will simply be the indexes for the entire dataset.
        if dataset.is_iterable():
            ids = list(dataset.ids())
            indexes = list(range(len(ids)))
        else:
            indexes = list(range(len(dataset)))

        # Note that when sampler=None, a default sampler is used. The default config
        # defines shuffle=False, which should prevent any shuffling of of the data.
        # We expect that this will be the primary use case when running inference.
        return idist.auto_dataloader(dataset, sampler=None, **data_loader_kwargs), indexes

    # Sanitize split argument
    if isinstance(split, str):
        split = [split]

    # Configure the torch rng
    torch_rng = torch.Generator()
    seed = config["data_set"]["seed"] if config["data_set"]["seed"] else None
    if seed is not None:
        torch_rng.manual_seed(seed)

    if dataset.is_iterable():
        ids = list(dataset.ids())
        indexes = list(range(len(ids)))
        dataloaders = {
            s: (idist.auto_dataloader(dataset, pin_memory=True, **data_loader_kwargs), indexes) for s in split
        }
    else:
        # Create the indexes for all splits based on config.
        indexes = create_splits(dataset, config)

        # Create samplers and dataloaders for each split we are interested in
        samplers = {s: SubsetSequentialSampler(indexes[s]) if indexes.get(s) else None for s in split}

        dataloaders = {
            split: (idist.auto_dataloader(dataset, sampler=sampler, **data_loader_kwargs), indexes[split])
            if sampler
            else None
            for split, sampler in samplers.items()
        }

        none_keys = [k for k, v in dataloaders.items() if v is None]
        for key in none_keys:
            del dataloaders[key]

    # Return only one if we were only passed one split in, return the dictionary otherwise.
    return dataloaders[split[0]] if len(split) == 1 else dataloaders


def create_splits(data_set: Dataset, config: dict):
    """Returns train, test, and validation indexes constructed to be used with the passed in
    dataset. The allocation of indexes in the underlying dataset to samplers depends on
    the data_set section of the config dict.

    Parameters
    ----------
    data_set : Dataset
        The data set to use
    config : dict
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


# ! Need to go through and clean up the variables here. I think `device` and `engine`
# ! are not used, but we'll need to double check before pulling out all the wiring.
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
        batch = tuple(torch.from_numpy(i).to(device) for i in batch)
    else:
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
    torch.set_default_device(device.type)
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
    model = idist.auto_model(model)
    evaluator = create_engine("infer_batch", device, model, config)

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
    model = idist.auto_model(model)
    tensorboardx_logger = get_tensorboard_logger()

    validator = create_engine("validate_batch", device, model, config)
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
    model = idist.auto_model(model)
    tensorboardx_logger = get_tensorboard_logger()

    tester = create_engine("test_batch", device, model, config)
    fixup_engine(tester)

    @tester.on(Events.STARTED)
    def set_model_to_eval_mode():
        model.eval()

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

    return tester


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
    model = idist.auto_model(model)
    trainer = create_engine("train_batch", device, model, config)
    tensorboardx_logger = get_tensorboard_logger()
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

    @trainer.on(Events.COMPLETED)
    def attach_final_metrics_to_model(trainer):
        # Attach the final training metrics to the model object for easy access
        model.final_training_metrics = trainer.state.output

    pbar = ProgressBar(persist=False, bar_format="")
    pbar.attach(trainer)

    return trainer


def create_save_batch_callback(dataset, results_dir):
    """Create a callback function for saving batch results during inference or testing.

    This factory function creates a closure that captures the dataset, indexes, and output
    directory, then returns a callback that can be used with create_evaluator to save
    model outputs batch by batch.

    Parameters
    ----------
    dataset : Dataset
        The dataset being processed (must have an ids() method)
    results_dir : Path
        Directory where results should be saved

    Returns
    -------
    callable
        A callback function with signature (batch, batch_results) that saves results
    """
    from hyrax.data_sets.inference_dataset import InferenceDataSetWriter

    data_writer = InferenceDataSetWriter(dataset, results_dir)
    write_index = 0

    def _save_batch(batch: Union[torch.Tensor, list, tuple, dict], batch_results: torch.Tensor):
        """Receive and write batch results to results_dir immediately."""
        nonlocal write_index
        nonlocal data_writer

        batch_len = len(batch_results)

        # Ensure the batch results are on CPU and detached from the computation graph
        batch_results = batch_results.detach().to("cpu")

        batch_object_ids = batch["object_id"]

        # Ensure that everything to be written is in numpy format, and write it out
        data_writer.write_batch(np.array(batch_object_ids), [t.numpy() for t in batch_results])
        write_index += batch_len

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
