import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import Any

trace_result = None
logger = logging.getLogger(__name__)


def trace_dataset_func(func=None, *, params_to_capture=None, result_name=None, stage_name="dataset_getter"):
    """
    Decorator to add tracing to a custom dataset function. By default captures all parameters and
    return value placing the function in the 'dataset_getter' stage.
    """
    params_to_capture = {} if params_to_capture is None else params_to_capture
    return trace_func(
        func, params_to_capture=params_to_capture, result_name=result_name, stage_name=stage_name
    )


def trace_model_func(func=None, *, params_to_capture=None, result_name=None, stage_name="evaluation"):
    """
    Decorator to add tracing to a custom model function. By default captures all parameters and
    return value placing the function in the 'evaluation' stage.
    """
    params_to_capture = {} if params_to_capture is None else params_to_capture
    return trace_func(
        func, params_to_capture=params_to_capture, result_name=result_name, stage_name=stage_name
    )


def trace_func(func=None, *, params_to_capture=None, result_name=None, stage_name):
    """
    Generic decorator to trace a user-defined function in a particular stage.

    The name of a Trace Result stage must be provided to use this decorator directly.
    """
    params_to_capture = {} if params_to_capture is None else params_to_capture

    def decorate(func):
        return TraceResult._make_shim(
            func,
            TraceDef(
                disp_name=func.__name__,
                func_name=func.__name__,
                params_to_capture=params_to_capture,
                result_name=result_name,
                stage_name=stage_name,
            ),
        )

    return decorate if func is None else decorate(func)


def trace_verb_data(verb_run_func):
    """
    Simple wrapper decorator for verbs to implement the trace=<num data items> interface

    This decorator:
    1. Adds a keyword argument "trace" which takes a number controlling how many data items
       are run through the verb. This is accomplished by hyrax config modification and by shimming
       particuar DataProvider methods

    2. Allows the verb's return value to be passed through, but in the case of trace being set
       the verb returns a TraceResult object that captures the order, parameter values, and return values
       of the main steps in Hyrax's default data pipeline for debugging purposes.

    """

    @wraps(verb_run_func)
    def trace_wrapper(self, *args, **kwargs):
        global trace_result
        trace = kwargs.pop("trace", None)
        with TraceContext(trace, self.config) as modified_config:
            self.config = modified_config
            retval = verb_run_func(self, *args, **kwargs)
            return trace_result if trace else retval

    return trace_wrapper


class TraceContext:
    """
    In order to trace we: 1) shim class methods and 2) modify hyrax config.

    Due to the class-level shims it is absolutely vital that even during exception handling we are able to
    remove these shims. This removal returns classes to their pre-trace state and keeps the effects of
    the shimming contained to the runtime of a single verb in a long-running notebook.

    Therefore verbs using data tracing should use the @trace_data decorator or implement the pattern below

    with TraceContext(trace, self.config) as modified_config:
        self.config = modified_config

        ...verb code...

        return get_trace() if trace else retval

    """

    def __init__(self, trace_arg: Any, config):
        self.trace_arg = trace_arg
        self.config = config

    def __enter__(self):
        # Only do something if we are running a trace.
        if self.trace_arg:
            global trace_result
            logger.warning("Starting Trace")
            trace_batch_size = self.trace_arg if isinstance(self.trace_arg, int) else 1
            logger.warning(f"Trace mode enabled, will only run a single batch of length {trace_batch_size}")
            trace_result = TraceResult(trace_batch_size)

            # We set global configs to cause an early return that traces a small number of data pts.
            self.config["train"]["epochs"] = 1
            self.config["data_loader"]["batch_size"] = trace_batch_size

            # Having cache running means some calls to getters aren't captured in a typical capture,
            # because those same data were fetched during pre-flight, so we turn caching off on a
            # trace to prevent this effect
            #
            # TODO: Rolling tracing. We will need a different solution here if we are ever running
            #       the trace as a ring buffer, holding essentially the "last batch" up to a crash.
            #       This only works because tracing implies an incredibly short run of data.
            self.config["data_set"]["use_cache"] = False

        return self.config

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleanup hooks regardless of any exception passed.
        global trace_result
        if trace_result:
            trace_result.remove_class_level_shims()

        trace_result = None


def get_trace():
    """Get the current global trace results object. Returns None if no trace is active"""
    global trace_result
    return trace_result


def reset_trace():
    """
    Reset the current global trace results object, removing all captured data
    Valid to call whether trace is active.
    """
    global trace_result
    if trace_result:
        logger.debug("Resetting Trace Results")
        trace_result.reset()


@dataclass
class TraceDef:
    """
    A record that needs to be filled out whenever a function is instrumented for tracing in TraceResult

    Contains values that must be passed through TraceResult.instrument_*, TraceResult.make_shim,
    and TraceResult.trace_call in order that TraceResults are legible when printed.
    """

    disp_name: str
    func_name: str
    params_to_capture: dict[str, int]
    result_name: str
    stage_name: str


class TracePrintable(ABC):
    """
    Base class defining foundational behavior for TraceResult, TraceStage, and TraceCall which are the
    user-accessible and building blocks of a trace.

    Child classes must implement __str__ for printing and __getitem__ for inspection.
    """

    def __repr__(self):
        """
        __repr__ and __str__ mean the same thing. This goes against python philosophy on __repr__ being
        essentially a serialized string of the class; however notebooks call __repr__ to display objects,
        and we would like the __str__ code to have correct connotation for robots an humans viewing the
        code through a peephole. That is: __str__ means "Human readable and perhaps incomplete representation"
        """
        return str(self)

    def __getattr__(self, attr):
        """
        __getattr__ always calls getitem. This implements the notion that if you get a trace object in
        a notebook, you ought to be able to equally well say trace_result["evaluation"] and
        trace_result.evaluation to ask for just the function calls in the evaluation stage. The intent is
        to make it so that any attempt by the user to look inside the class routes to the things they probably
        want.
        """
        return self[attr]

    def __dir__(self):
        """
        Force implementation of __dir__ on subclasses to direct typeahead in notebook environments toward
        valid identifiers within the trace.
        """
        return self._valid_keys()

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def _valid_keys(self):
        pass


class TraceResult(TracePrintable):
    """
    Result of a hyrax data tracing run, returned from certain data-handling verbs when trace=<non-zero number>
    is passed in a notebook.

    This object represents a small set of calls intended to track a handful of data values through the
    entire hyrax data processing pipeline in order to enable debugging of data issues.

    This object is meant to be printed out in a notebook, and contains multiple stages that are accessible
    using either trace_result.stage_name or trace_result["stage_name"] syntax.

    1. "dataset_getter" stage
    In this stage the HyraxQL getter functions on whatever datasets are in use are called. If you implemented
    a custom dataset, these are functions you wrote. Any dataset class functions decorated with
    @trace_dataset_func also have calls reported in this stage

    2. "resolve_data" stage
    In this stage DataProvider.resolve_data combines the results of the individual data getters into
    data dictionaries which each contain all requested columns for each datum

    3. "collate" stage
    Each data column is combined into a single batch tensor in this stage. If your dataset defines a custom
    collate function (e.g. time-series data with different lengths) it will be evaluated in this stage.
    Any NaN handling that is configured into hyrax is also performed in this stage.

    4. "prepare_inputs" stage
    The ML Model's prepare_inputs function is called in this stage and converts the data dictionary
    containing each column of batched tensor data into a single batch tensor that will form the input to
    the model's evaluation functions. If the model is doing supervized learning, the output will be
    a tuple of numpy arrays (inputs_0, [inputs_1, ..., inputs_n], labels)

    5. "evaluation" stage
    The ML model is evaluated or the training loop is run. Functions will be functions run on the model
    during this process, including `train_batch` , `forward` and similar. If you implemented a custom model
    you wrote these functions. Any model functions decorated with @trace_model_func will also show up here.

    """

    def __init__(self, trace_batch_size: int):
        self.shimmed_funcs = []
        self.trace_batch_size = trace_batch_size

        # xcxc todo make capturing make sense for "engine" verb
        # which doesn't have quite the same model structure
        from hyrax.models.model_registry import MODEL_REGISTRY

        for model_cls in MODEL_REGISTRY.values():
            for name in dir(model_cls):
                if callable(getattr(model_cls, name, None)):
                    trace_def = None
                    if name == "forward" or name == "infer_batch":
                        trace_def = TraceDef(
                            disp_name=f"{model_cls.__name__}__{name}",
                            func_name=name,
                            # _make_shim forwards all args to trace_call, so class-level instance
                            # methods include `self` at arg 0.
                            params_to_capture={"batch": 1},
                            result_name="batch_results",
                            stage_name="evaluation",
                        )
                    if name == "train_batch" or name == "validate_batch" or name == "test_batch":
                        trace_def = TraceDef(
                            disp_name=f"{model_cls.__name__}__{name}",
                            func_name=name,
                            # _make_shim forwards all args to trace_call, so class-level instance
                            # methods include `self` at arg 0.
                            params_to_capture={"batch": 1},
                            result_name="loss_dict",
                            stage_name="evaluation",
                        )
                    if name == "prepare_inputs":
                        trace_def = TraceDef(
                            disp_name=f"{model_cls.__name__}__{name}",
                            func_name=name,
                            params_to_capture={"batch_dict": 0},
                            result_name="batch_ndarray",
                            stage_name="prepare_inputs",
                        )
                    if trace_def:
                        self.instrument_class_data_handler(model_cls, trace_def)

        # Drop the length of the dataprovider so we end train/inference/test/engine runs early
        from hyrax.datasets.data_provider import DataProvider

        self.reduce_len(DataProvider)

        # Clear our representation of calls.
        self.reset()

    def reset(self):
        """Reset the Trace Result object to having no calls"""
        # Static list of stages to start
        self.stages = {
            "dataset_getter": TraceStage(),
            "resolve_data": TraceStage(),
            "collate": TraceStage(),
            "prepare_inputs": TraceStage(),
            "evaluation": TraceStage(),
        }

    def __getitem__(self, ref):
        return self.stages[ref]

    def _valid_keys(self):
        return list(self.stages.keys())

    def reduce_len(self, cls):
        """
        Inserts a len method which reduces the length of the passed in class in order to
        accommodate early return in trace mode.

        This is necessary because hyrax does not control the main loop of inference/training
        for most ML verbs, so the layer that does control it must get an appropriate stop condition
        from Hyrax's data structures
        """
        raw_func = cls.__dict__.get("__len__")

        def new_len(obj):
            import numpy as np

            # We actually need the length to be one-past-the-end of whever split index we will
            # encounter at the end of the first (and only) batch
            #
            # This accommodates the situation where there is a split_fraction defined in the data
            # definition.
            if obj.split_indices is not None:
                return obj.split_indices[self.trace_batch_size - 1] + 1

            split_fraction = 1.0 if obj.split_fraction is None else obj.split_fraction
            max_len = int(np.ceil(self.trace_batch_size / split_fraction))

            # Don't ever make the new length longer than the old length
            # Can happen in some weird split situations on small datasets (like RandomDataset)
            # in testing contexts
            return min(max_len, raw_func(obj))

        cls.__len__ = new_len
        self.shimmed_funcs.append((cls, "__len__", raw_func))

    def remove_class_level_shims(self):
        """
        Clean up all of our class level shims. This should happen when verbs exit
        even if via exception. See TraceContext for the mechanism by which this is achieved.
        """
        logger.debug("Removing class level shims")
        for cls, func_name, original_member in self.shimmed_funcs:
            setattr(cls, func_name, original_member)

    def trace_call(self, trace_def: TraceDef, *args):
        """
        This is the main location where data is collected. Shim functions call this method in order to
        log to the trace that a call to the shimmed function has occurred.

        We capture parameters and return value here.
        """
        logger.debug(f"Received Trace {trace_def.stage_name} {trace_def.func_name}")

        captured_params = {}
        if len(trace_def.params_to_capture) == 0:
            for index, arg in enumerate(args):
                name = f"{index:0>3}_call"
                captured_params[name] = arg
        else:
            for param_name, param_idx in trace_def.params_to_capture.items():
                if isinstance(param_idx, int):
                    captured_params[param_name] = args[param_idx]
                else:
                    raise RuntimeError("Captured trace params must always be integer args, not kwargs")
                    # TODO if we need this, pass kwargs into trace_call and figure out the schema for
                    # TraceDef's to define the keyword to pull.

        result_name = trace_def.result_name if trace_def.result_name else "return_value"

        call_record = TraceCall(
            disp_name=trace_def.disp_name,
            func_name=trace_def.func_name,
            params=captured_params,
            # These will be filled in by update_retval, see below
            retval={result_name: None},
            duration_ns=None,  # Xcxc make durations happen
        )

        self.stages[trace_def.stage_name].append(call_record)

        # This lets caller fill in return value and duration
        # This is necessary for trace readability because call order != return order, especially when i/o
        # is involved. For many of these functions, so we want all our lists above to capture the order
        # the functions were called in, because that is most user-legible.
        def update_retval(retval, duration_ns):
            call_record.retval[result_name] = retval
            call_record.duration_ns = duration_ns

        return update_retval

    def instrument_prepare_inputs(self, model):
        """
        Instrument the prepare_inputs function on an instance of a model. This occurs when we load the
        model and will be using a prepare_inputs function which was attached to the model by
        hyrax machinery (@hyrax_model). This is usually a old to_tensor function, a loaded prepare_inputs
        function from a checkpoint or our default prepare_inputs function.

        Note: Class level shimming of prepare_inputs occurs in the constructor and covers the case where
        the model class defines prepare_inputs directly
        """
        prepare_inputs_fn = model.prepare_inputs
        trace_def = TraceDef(
            disp_name=f"{model.__class__.__name__}_inst_prepare_inputs",
            func_name="prepare_inputs",
            params_to_capture={"batch_dict": 0},
            result_name="batch_ndarray",
            stage_name="prepare_inputs",
        )
        return self.instrument_instance_data_handler(model, prepare_inputs_fn, trace_def)

    def instrument_prepare_inputs_fn(self, prepare_inputs_fn):
        """
        Instrument the prepare_inputs function on a bare function. This is used in the engine verb
        when we don't have a pytorch model class to attach to.
        """
        trace_def = TraceDef(
            disp_name="saved__prepare_inputs",
            func_name="prepare_inputs",
            params_to_capture={"batch_dict": 0},
            result_name="batch_ndarray",
            stage_name="prepare_inputs",
        )
        return self._make_shim(prepare_inputs_fn, trace_def)

    def instrument_dataset_getter(self, dataset, getter, friendly_name, field_name):
        """
        Instrument a dataset get_* function. Called by DataProvider to insert shims before
        any betters are called
        """
        trace_def = TraceDef(
            disp_name=f"{friendly_name}__get_{field_name}",
            func_name=f"get_{field_name}",
            params_to_capture={"index": 1},
            result_name=field_name,
            stage_name="dataset_getter",
        )
        return self.instrument_instance_data_handler(dataset, getter, trace_def)

    def instrument_dataset_collate(self, dataset, collate_fn, friendly_name):
        """
        Instrument a dataset collate function. Also called by DataProvider to insert shims
        into all the custom dataset collate functions it finds during dataset preparation.
        """
        trace_def = TraceDef(
            disp_name=f"{friendly_name}__collate",
            func_name="collate",
            params_to_capture={"samples": 0},
            result_name="batch_dict",
            stage_name="collate",
        )
        return self.instrument_instance_data_handler(dataset, collate_fn, trace_def)

    def instrument_dataprovider(self, dataprovider):
        """
        Instrument the various data handling functions in DataProvider.

        We use instance level shims here
        """
        self.instrument_instance_data_handler(
            dataprovider,
            dataprovider.resolve_data,
            TraceDef("DataProvider__resolve_data", "resolve_data", {"index": 1}, "data_dict", "resolve_data"),
        )
        self.instrument_instance_data_handler(
            dataprovider,
            dataprovider.collate,
            TraceDef("DataProvider__collate", "collate", {"batch_dicts": 1}, "batch_dict", "collate"),
        )
        self.instrument_instance_data_handler(
            dataprovider,
            dataprovider.handle_nans,
            TraceDef(
                "DataProvider__handle_nans", "handle_nans", {"batch_dict": 1}, "batch_dict_no_nan", "collate"
            ),
        )

    def instrument_engine_verb(self, engine_verb):
        """
        Instrument the various data handling functions in the engine verb.

        These are instance level shims, because by the time we know whether a verb is tracing or not, the
        verb class instance has already been constructed, so we must operate on the instance.
        """
        self.instrument_instance_data_handler(
            engine_verb,
            engine_verb.create_ort_inputs,
            TraceDef(
                "Engine__create_ort_inputs",
                "create_ort_inputs",
                {"prepared_batch": 1},
                "ort_inputs",
                "evaluation",
            ),
        )
        self.instrument_instance_data_handler(
            engine_verb,
            engine_verb.run_onnx_batch,
            TraceDef(
                "Engine__run_onnx_batch", "run_onnx_batch", {"ort_inputs": 1}, "onnx_results", "evaluation"
            ),
        )

    def instrument_instance_data_handler(self, obj, original_member, trace_def: TraceDef):
        """
        Inserts trace instrumentation on a method of a python class instance.

        DOES NOT WORK ON classes, see instrument_class_data_handler.

        Parameters
        ----------
        obj : class instance
            The instance of the object that has the member function we are shimming
        original_member : callable
            The callable we are shimming. Obtain via ``obj.method_name`` or
            ``getattr(obj, "method_name")``.
        trace_def : TraceDef
            A TraceDef defining what we're tracing from this function.

        Returns
        -------
        callable
            The shim callable that has been set on ``obj`` at ``trace_def.func_name``.
        """
        logger.debug(f"Shimming {obj.__class__.__name__}.{trace_def.func_name}")
        shim = self._make_shim(original_member, trace_def)
        setattr(obj, trace_def.func_name, shim)
        return shim

    def instrument_class_data_handler(self, cls, trace_def: TraceDef):
        """
        Inserts trace instrumentation on a method of a python class.

        The shimmed method is placed on the class and is not returned.

        DOES NOT WORK ON class instances, see instrument_instance_data_handler.

        Parameters
        ----------
        cls : class
            The class that has the member function we are shimming
        trace_def : TraceDef
            A TraceDef defining what we're tracing from this function.

        Returns
        -------
            None
        """
        logger.debug(f"Shimming {cls.__name__}.{trace_def.func_name}")
        class_dict_member = cls.__dict__.get(trace_def.func_name, None)
        trace_shim = self._make_shim(class_dict_member, trace_def)
        setattr(cls, trace_def.func_name, trace_shim)

        # This is so we can remove the class-level shims out when we're done.
        self.shimmed_funcs.append((cls, trace_def.func_name, class_dict_member))

    @staticmethod
    def _make_shim(original_func, trace_def: TraceDef):
        """Make a shim function for the instrument_* functions to use.

        Parameters
        ----------
        original_func : callable
            The function (or bound method) being shimmed.
        trace_def : TraceDef
            Describes what data to capture during the call.
        """

        @wraps(original_func)
        def trace(*args, **kwargs):
            import time

            trace_obj = get_trace()

            if trace_obj:
                update_retval = trace_obj.trace_call(trace_def, *args)
                start_ns = time.monotonic_ns()

            func = (
                original_func.__func__
                if isinstance(original_func, (classmethod, MethodType))
                else original_func
            )

            retval = func(*args, **kwargs)

            if trace_obj:
                end_ns = time.monotonic_ns()
                update_retval(retval, end_ns - start_ns)

            return retval

        if isinstance(original_func, staticmethod):
            return staticmethod(trace)
        elif isinstance(original_func, classmethod):
            return classmethod(trace)
        elif isinstance(original_func, MethodType):
            return MethodType(trace, original_func.__self__)

        return trace

    def __str__(self):
        """
        Print out the stages of the trace.
        """
        repr = "Trace Stages {\n"
        for stage_name, trace_stage in self.stages.items():
            repr += f"\t{stage_name}: "
            repr += str(trace_stage).replace("\n", "\n\t")
            repr += "\n"
        repr += "}\n"
        return repr


class TraceStage(TracePrintable):
    """
    This is a container that holds a list of TraceCalls in order of execution, representing an entire
    stage of a TraceResult.

    It is intended to be printed and examined from a notebook.

    It supports two modes of user access through [] / __getitem__:
      1) [] with a number gets access to a TraceCall by number
      2) [] with a function name gets access to all of those functions as a list[TraceCall]
    """

    def __init__(self):
        self.calls = []
        self.func_dict = {}

    def append(self, call_record):
        """
        Append a single call record to this stage.
        """
        # Save the call under all calls
        self.calls.append(call_record)

        # Save under the call record for the correct display name
        if self.func_dict.get(call_record.disp_name) is None:
            self.func_dict[call_record.disp_name] = []
        self.func_dict[call_record.disp_name].append(call_record)

    def __getitem__(self, idx_or_func_name):
        try:
            try:
                idx = int(idx_or_func_name)
                return self.calls[idx]
            except ValueError:
                pass
            return self.func_dict[idx_or_func_name]
        except (IndexError, KeyError) as e:
            msg = f"{idx_or_func_name} not found. You can ask for:\n"
            msg += f"a number < {len(self.calls)} to get a particular function call\n"
            msg += "a function name, to get a list of all the calls of that function\n"
            msg += "Valid function keys are:\n"
            for key in self.func_dict:
                msg += f"{key}\n"
            e.add_note(msg)
            raise

    def _valid_keys(self):
        return list(self.func_dict.keys())

    def __len__(self):
        return len(self.calls)

    def __str__(self):
        return f"[{self._repr_calls()}]" if len(self) > 0 else "[]"

    def _repr_calls(self):
        repr = "\n"
        for call in self.calls:
            all_calls = str(call).replace("\n", "\n\t")
            repr += f"\t{all_calls}\n"
        return repr


@dataclass(repr=False)  # Don't have @dataclass generate a repr, because it overrides TracePrintable's repr
class TraceCall(TracePrintable):
    """
    An individual function call that is part of a trace, Captures argument and return values of
    the given function, which are accessible via [] or . operators.

    This object is intended to be printed and examined from a notebook
    """

    disp_name: str
    func_name: str
    params: dict[str, Any]
    retval: dict[str, Any]
    duration_ns: float

    def __str__(self):
        params_repr = ", ".join(list(self.params.keys()))
        retval_repr = ", ".join(list(self.retval.keys()))

        duration_ms = float(self.duration_ns) / float(1_000_000)

        repr = f"{self.disp_name}({params_repr}) -> {retval_repr} duration={duration_ms:.3g} ms\n"

        repr += "inputs:\n"
        for param_name, param_value in self.params.items():
            value_str = self._repr_value(param_value).replace("\n", "\n\t")
            repr += f"\t{param_name} = {value_str}\n"

        repr += "outputs:\n"
        for param_name, param_value in self.retval.items():
            value_str = self._repr_value(param_value).replace("\n", "\n\t")
            repr += f"\t{param_name} = {value_str}\n"

        return repr

    def __getitem__(self, key):
        all_values = list(self.params.values()) + list(self.retval.values())
        try:
            try:
                idx = int(key)
                return all_values[idx]
            except ValueError:
                pass

            if self.params.get(key) is not None and self.retval.get(key) is not None:
                return self.params[key], self.retval[key]
            elif self.params.get(key) is not None:
                return self.params[key]
            elif self.retval.get(key) is not None:
                return self.retval[key]
        except (IndexError, KeyError) as e:
            msg = f"{key} not found in function parameters or return value\n"
            msg += f"You can access function parameters and return value by index < {len(all_values)}\n"
            msg += "Or by name where the names for this object are: \n"
            for name in self._valid_keys():
                msg += f"{name}\n"
            e.add_note(msg)
            raise

    def _valid_keys(self):
        # Note, not guaranteed unique, but should be most of the time since we have been
        # careful with names given to TraceDef() calls.
        return list(self.params.keys()) + list(self.retval.keys())

    def _repr_value(self, param_value):
        import numpy as np
        import torch

        atomic_types = set([int, float, bool, np.integer, np.bool_, np.floating])
        tensor_types = set([np.ndarray, torch.Tensor])

        repr = ""
        if isinstance(param_value, tuple(atomic_types)):
            repr += f"{param_value}"
        elif isinstance(param_value, str):
            repr += f"'{param_value}'"
        elif isinstance(param_value, list):
            repr += f"<{type(param_value).__name__} len={len(param_value)}> [\n"
            for list_elem in param_value:
                list_elem_str = self._repr_value(list_elem).replace("\n", "\n\t")
                repr += f"\t{list_elem_str}\n"
            repr += "]"
        elif isinstance(param_value, dict):
            repr += "{\n"
            for key, value in param_value.items():
                value_str = self._repr_value(value).replace("\n", "\n\t")
                repr += f"\t{key}: {value_str} \n"
            repr += "}"
        elif isinstance(param_value, tuple):
            repr += "(\n"
            for value in param_value:
                value_str = self._repr_value(value).replace("\n", "\n\t")
                repr += f"\t{value_str},\n"
            # repr =  repr[:-1] if len(param_value) > 1 else repr # no trailing comma on tuples with len > 1
            repr += ")"
        elif isinstance(param_value, tuple(tensor_types)):
            # Everything numpy is on cpu. If on GPU the torch-specific branch will update this
            device = "cpu"
            if isinstance(param_value, np.ndarray):
                type_name = "numpy.ndarray"
                if np.any(np.vectorize(lambda x: isinstance(x, (np.str_, str)))(param_value)):
                    as_torch = torch.from_numpy(np.vectorize(lambda x: hash(x))(param_value))
                else:
                    as_torch = torch.from_numpy(param_value)
            else:
                type_name = "torch.Tensor"
                device = param_value.device
                # Have to pull to CPU to perform hash calc
                as_torch = param_value.to("cpu")

            hash_val = as_torch.hash_tensor()

            shape = tuple(param_value.shape)
            # type_name = type(param_value).__name__
            repr += f"<{type_name} shape={shape} hash={hash_val} device={device}>"
        else:
            repr += f"UNSUPPORTED TYPE {type(param_value)}"

        return repr
