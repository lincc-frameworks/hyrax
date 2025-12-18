import logging

import onnx
import onnxruntime
from numpy import allclose

logger = logging.getLogger(__name__)


def export_to_onnx(model, sample, config, ctx):
    """Dispatching function to convert a ML framework model into an ONNX model.

    Parameters
    ----------
    model : ML framework model
        The model that was just trained using the ML framework. i.e. PyTorch
    sample : Tensor
        This sample is the result of running a batch of data through the data
        loader and the model's `to_tensor` function. It is used to compare the
        output of the ONNX model against the output of the PyTorch model.
    config : dict
        The parsed config file as a nested dict
    ctx : dict
        A context dictionary containing info needed for the conversion to ONNX.
    """

    # build the output ONNX file path
    onnx_opset_version = config["onnx"]["opset_version"]
    onnx_output_filepath = ctx["results_dir"] / "model.onnx"

    # use the "ml_framework" context value to determine how to convert to ONNX.
    sample_out = None
    if ctx["ml_framework"] == "pytorch":
        sample_out = _export_pytorch_to_onnx(model, sample, onnx_output_filepath, onnx_opset_version)
    else:
        logger.warning("No ONNX export implementation for the given ML framework.")
        return

    # check the ONNX model for correctness
    try:
        onnx_model = onnx.load(onnx_output_filepath)
        onnx.checker.check_model(onnx_model)
    except:  # noqa E722
        logger.error(f"Failed to create ONNX model. {ctx['ml_framework']} implementation has been saved.")

    # Check the ONNX model against the PyTorch model. Note that `sample` was
    # converted to numpy array when the model was converted to ONNX
    ort_session = onnxruntime.InferenceSession(onnx_output_filepath)

    # Log the ONNX model input details for debugging
    log_ort_inputs = "\n".join(
        [f"Name: {i.name}, shape: {i.shape}, type: {i.type}" for i in ort_session.get_inputs()]
    )
    logger.debug(f"ONNX inputs - {log_ort_inputs}")

    # Create the inputs array for the ONNX model
    ort_inputs = {}
    # ! May need to change this if-statement, if we find that to_tensor should return
    # ! something other than a tuple for multiple inputs.
    if isinstance(sample, tuple):
        for i in range(len(sample)):
            if len(sample[i]):
                ort_inputs[ort_session.get_inputs()[i].name] = sample[i]
    else:
        ort_inputs = {ort_session.get_inputs()[0].name: sample}

    # Run the ONNX model
    ort_outs = ort_session.run(None, ort_inputs)

    # verify ONNX model inference produces results close to the the original model
    if not allclose(sample_out, ort_outs[0], rtol=1e-03, atol=1e-05):
        logger.warning("The outputs from the PyTorch model and the ONNX model are not close.")

    logger.debug(f"Exported model to ONNX format: {onnx_output_filepath}")


def _export_pytorch_to_onnx(model, sample, output_filepath, opset_version):
    """Specific implementation to convert PyTorch model to ONNX format. This uses
    the older (torch<2.9) export capabilities. And only supports up the opset
    version 20.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be converted to ONNX format.
    sample : NumPy array or list of NumPy arrays
        A sample of input data to the model. This is used to trace the model
        during the export process.
    output_filepath : Path
        The file path where the ONNX model will be saved.
    opset_version : int
        The ONNX opset version to use for the export.
    """

    # deferred import to reduce start up time
    import torch
    from torch.onnx import export
    from torch.utils.data.dataloader import default_convert

    # set model in eval mode and move it to the CPU to prep for export to ONNX.
    model.train(False)
    model.to("cpu")

    # set the default device to CPU and convert the sample to torch Tensors
    torch.set_default_device("cpu")
    torch_sample = default_convert(sample)

    # Run a single sample through the model. We'll check this against the output
    # from the ONNX version to make sure it's the same, i.e. `np.assert_allclose`.
    sample_out = model(torch_sample)

    # Here we attempt to identify inputs in torch_sample that have a variable
    # shape due to batch size and set those as dynamic axes in the ONNX model.
    input_names = []
    dynamic_axes = {}

    # torch_sample is returned from default_convert as either a single Tensor or
    # a list of Tensors.
    if isinstance(torch_sample, list):
        for i in range(len(torch_sample)):
            # For supervised models, the label or target should be empty
            # so we will not include those in the input names. Any labels should
            # be the last element in the list.
            if len(torch_sample[i]):
                input_names.append(f"input_{i}")
                dynamic_axes[f"input_{i}"] = {0: "batch_size"}
    else:
        input_names.append("input")
        dynamic_axes["input"] = {0: "batch_size"}

    # Output is assumed to always have a dynamic batch size.
    dynamic_axes["output"] = {0: "batch_size"}

    # export the model to ONNX format
    export(
        model,
        torch_sample,
        output_filepath,
        opset_version=opset_version,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,  # newer versions of torch will use dynamo by default
    )

    # Make sure that the output is on the CPU
    if sample_out.device.type != "cpu":
        sample_out = sample_out.to("cpu")

    # Return the output of the model as numpy array
    return sample_out.detach().numpy()


def _export_pytorch_to_onnx_v2(model, sample, output_filepath, opset_version):
    """Currently unused.
    Specific implementation to convert PyTorch model to ONNX format using
    torch Dynamo export capabilities.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be converted to ONNX format.
    sample : NumPy array or list of NumPy arrays
        A sample of input data to the model. This is used to trace the model
        during the export process.
    output_filepath : Path
        The file path where the ONNX model will be saved.
    opset_version : int
        The ONNX opset version to use for the export.
    """

    # deferred import to reduce start up time
    import torch
    from torch.onnx import export
    from torch.utils.data.dataloader import default_convert

    # set model in eval mode and move it to the CPU to prep for export to ONNX.
    model.train(False)
    model.to("cpu")

    # set the default device to CPU and convert the sample to torch Tensors
    torch.set_default_device("cpu")
    torch_sample = default_convert(sample)

    # Run a single sample through the model. We'll check this against the output
    # from the ONNX version to make sure it's the same, i.e. `np.assert_allclose`.
    sample_out = model(torch_sample)
    # Make sure that the output is on the CPU, detached, and as a numpy array
    sample_out = sample_out.to("cpu").detach().numpy()

    dynamic_shapes = []
    batch = torch.export.Dim("batch")

    # TODO: This should be built dynamically based on the structure of torch_sample.
    dynamic_shapes = [[{0: batch}, {0: batch}, {}]]

    export(
        model,
        (torch_sample,),  # exporter expects a tuple of inputs for `forward`
        output_filepath,
        opset_version=opset_version,
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        verbose=True,
        report=True,
        dump_exported_program=True,
        artifacts_dir=output_filepath.parent,
        input_names=["input"],
        output_names=["output"],
    )

    return sample_out
