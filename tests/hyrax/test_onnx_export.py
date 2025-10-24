"""Tests for ONNX export functionality"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from hyrax.model_exporters import _export_pytorch_to_onnx


class ModelWithTupleInput(nn.Module):
    """A simple model that expects a tuple of 3 tensors as input"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        """Expects x to be a tuple of (tensor1, tensor2, tensor3)"""
        if not isinstance(x, tuple) or len(x) != 3:
            raise TypeError(f"Expected tuple of 3 elements, got {type(x)}")
        # Use only the first tensor for simplicity
        tensor1 = x[0]
        return self.linear(tensor1)


class ModelWithSingleInput(nn.Module):
    """A simple model that expects a single tensor as input"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        """Expects x to be a single tensor"""
        if isinstance(x, tuple):
            raise TypeError("Expected a single tensor, got tuple")
        return self.linear(x)


def test_export_model_with_tuple_input():
    """Test that we can export a model that expects a tuple input to ONNX"""
    model = ModelWithTupleInput()
    model.eval()

    # Create a sample input tuple with 3 tensors
    sample = (
        torch.randn(2, 10),
        torch.randn(2, 5),
        torch.randn(2, 3),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "model.onnx"

        # This should not raise an error
        _export_pytorch_to_onnx(model, sample, output_path, opset_version=11)

        # Verify the ONNX file was created
        assert output_path.exists()


def test_export_model_with_single_tensor_input():
    """Test that we can export a model that expects a single tensor to ONNX"""
    model = ModelWithSingleInput()
    model.eval()

    # Create a sample input (single tensor)
    sample = torch.randn(2, 10)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "model.onnx"

        # This should not raise an error
        _export_pytorch_to_onnx(model, sample, output_path, opset_version=11)

        # Verify the ONNX file was created
        assert output_path.exists()
