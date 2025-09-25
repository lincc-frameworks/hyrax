import logging
from pathlib import Path

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
        A single sample from the training data loader. This is used to check the
        output of the ONNX model against the output of the PyTorch model.
    config : dict
        The parsed config file as a nested dict
    ctx : dict
        A context dictionary containing info needed for the conversion to ONNX.
    """

    # build the output ONNX file path
    model_filename = Path(config["train"]["weights_filename"]).stem
    onnx_opset_version = config["onnx"]["opset_version"]
    onnx_model_filename = f"{model_filename}_opset_{onnx_opset_version}.onnx"
    onnx_output_filepath = ctx["results_dir"] / onnx_model_filename

    # use the "ml_framework" context value to determine how to convert to ONNX.
    sample_out = None
    if ctx["ml_framework"] == "pytorch":
        sample, sample_out = _export_pytorch_to_onnx(model, sample, onnx_output_filepath, onnx_opset_version)
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
    ort_inputs = {ort_session.get_inputs()[0].name: sample}
    ort_outs = ort_session.run(None, ort_inputs)

    # verify ONNX model inference produces results close to the the original model
    if not allclose(sample_out, ort_outs[0], rtol=1e-03, atol=1e-05):
        logger.warning("The outputs from the PyTorch model and the ONNX model are not close.")

    logger.debug(f"Exported model to ONNX format: {onnx_output_filepath}")


def _export_pytorch_to_onnx(model, sample, output_filepath, opset_version):
    """Specific implementation to convert PyTorch model to ONNX format. This
    function will also:
    -  Run `sample` through the model before converting the model to ONNX
    -  Convert `sample` to a numpy array
    """

    # deferred import to reduce start up time
    from torch.onnx import export

    # set model in eval mode and move it to the CPU to prep for export to ONNX.
    model.train(False)
    model.to("cpu")

    # run a single sample through the model. We'll check this against the output
    # from the ONNX version to make sure it's the same, i.e. `np.assert_allclose`.
    sample_out = model(sample)

    # export the model to ONNX format
    export(
        model,
        sample,
        output_filepath,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # return the input sample as a numpy array and the output of the sample run
    # through the model as numpy array
    return sample.numpy(), sample_out.detach().numpy()
