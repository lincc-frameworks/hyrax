# ONNX Export Limitations for Supervised Models

**Status**: Investigation complete. No short-term solution available using torch.jit methods.

## Issue
ONNX export currently fails for supervised models that use `(data, label)` tuples during training.

## Current Behavior
When attempting to export a supervised model to ONNX, the export fails at `model_exporters.py:102` with:
```
AttributeError: 'tuple' object has no attribute 'numpy'
```

This occurs because:
1. Supervised models use `to_tensor()` which returns `(image_tensor, label_tensor)` 
2. The model's `forward()` handles this correctly (extracts only the data)
3. But the export code assumes `sample` is a simple tensor and calls `sample.numpy()`

## Investigation Results

### Torch.jit.script Approach
- **Status**: Not viable as a short-term solution
- **Issue**: ScriptModules cannot be exported directly to ONNX
- Requires conversion via `TS2EPConverter` which adds significant complexity

### Torch.jit.trace Approach  
- **Status**: Not compatible with tuple inputs
- **Issue**: `torch.jit.trace` unpacks tuples automatically, calling `forward(data, label)` instead of `forward((data, label))`
- This breaks models that expect a single tuple argument

## Root Cause
ONNX's tracing-based export mechanism records only the operations that were actually executed. For supervised models where the `label` is not used in the `forward()` pass, the label input will always be pruned from the exported model.

## Test Coverage
See `tests/hyrax/test_to_onnx.py` for:
- `test_to_onnx_supervised_export`: Documents the current failure
- `test_to_onnx_supervised_export_with_jit_script`: Tests torch.jit.script approach
- `test_to_onnx_supervised_export_with_jit_trace`: Tests torch.jit.trace approach

## Recommendations
1. **Short-term**: Document this limitation for users
2. **Medium-term**: Consider separate inference-only models that don't require labels
3. **Long-term**: Explore custom ONNX export using PyTorch 2.x `torch.export` API with explicit input specifications
