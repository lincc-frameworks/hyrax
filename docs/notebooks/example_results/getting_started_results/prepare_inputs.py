@staticmethod
def prepare_inputs(data_dict) -> tuple:
    """Does NOT convert to PyTorch Tensors.
    This works exclusively with numpy data types and returns
    a tuple of numpy data types."""

    import numpy as np

    if "data" not in data_dict:
        raise RuntimeError("Unable to find `data` key in data_dict")

    data = data_dict["data"]
    image = np.asarray(data["image"], dtype=np.float32)
    label = np.asarray(data.get("label", []), dtype=np.int64)

    return (image, label)
