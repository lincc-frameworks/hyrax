from collections.abc import Iterable
from typing import Any

import numpy as np
from torch.utils.data._utils.collate import default_collate


def hyrax_collate(batch: Iterable[dict]) -> dict:
    """Custom collate function for Hyrax that pads unequal-length numpy array fields.

    The function recursively walks the structure of the sample dictionaries and:
    - For numpy arrays of varying shape along the batch dimension, pads them to the
      elementwise maximum shape and returns a tuple (padded_array, pad_mask) where
      pad_mask is boolean with True indicating padded elements.
    - For other types, defers to torch.utils.data._utils.collate.default_collate.

    Returns a single collated dict (not a list) where each top-level friendly-name maps
    to its collated structure.
    """

    samples = list(batch)
    if len(samples) == 0:
        return {}

    def _recursive_collate(values: list[Any]) -> Any:
        """Collate a list of values that correspond to the same field across samples."""
        first = values[0]

        # dict -> recurse per-key
        if isinstance(first, dict):
            out: dict[str, Any] = {}
            keys = first.keys()
            for k in keys:
                field_vals = [v[k] for v in values]
                out[k] = _recursive_collate(field_vals)
            return out

        # numpy arrays -> pad & return (padded_stack, mask_stack)
        if isinstance(first, np.ndarray):
            padded, mask = pad_to_max_shape(values)
            return padded, mask

        # list/tuple -> check contents; if elements are arrays, try to pad them
        if isinstance(first, (list, tuple)):
            # all elements are numpy arrays -> pad them
            if all(isinstance(v, np.ndarray) for v in values):
                padded, mask = pad_to_max_shape(values)
                return padded, mask
            # otherwise, let default_collate handle a sequence of sequences/scalars
            return default_collate(values)

        # Fallback: let default_collate handle tensors, scalars, strings, etc.
        return default_collate(values)

    # Top-level: batch is list of sample dicts. Each sample has friendly-name keys.
    first_elem = samples[0]
    if isinstance(first_elem, dict):
        collated: dict[str, Any] = {}
        for friendly in first_elem:
            vals = [s[friendly] for s in samples]
            collated[friendly] = _recursive_collate(vals)
        return collated

    # If batch elements are not dicts, just recurse on the list directly
    return _recursive_collate(samples)


def pad_to_max_shape(arrays: Iterable[np.ndarray], pad_value=0) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad a list of numpy arrays so all have the same shape (the elementwise max shape).
    Pads at the end (after) for each axis, preserves dtype, and returns a stacked array
    with shape (N, *max_shape) and a boolean mask of the same shape where True indicates
    elements that were added as padding.

    Parameters
    ----------
    arrays : iterable of np.ndarray
        All arrays must have the same number of dimensions.
    pad_value : scalar
        Value used to fill the padded area.

    Returns
    -------
    (padded_stack, mask_stack) : tuple
        padded_stack : np.ndarray, shape (N, *max_shape)
            The stacked, padded arrays.
        mask_stack : np.ndarray (bool), shape (N, *max_shape)
            Boolean mask where True indicates the element is padding (not present
            in the original array).
    """
    arrays = list(arrays)
    if len(arrays) == 0:
        return np.array([]), np.array([])

    # verify same ndim
    nd = arrays[0].ndim
    for a in arrays:
        if a.ndim != nd:
            raise ValueError("All arrays must have the same number of dimensions")

    # determine elementwise maximum shape
    max_shape = tuple(max(a.shape[d] for a in arrays) for d in range(nd))

    padded: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for a in arrays:
        dtype = a.dtype
        # compute pad widths: (before, after) for each axis — we pad only after
        pad_width = tuple((0, max_shape[d] - a.shape[d]) for d in range(nd))
        # ensure pad_value cast to target dtype to avoid upcast surprises
        pv = np.asarray(pad_value, dtype=dtype)
        p = np.pad(a, pad_width=pad_width, mode="constant", constant_values=pv)
        padded.append(p)

        # build mask: False where padding was added
        full_mask = np.zeros(max_shape, dtype=bool)
        original_slices = tuple(slice(0, a.shape[d]) for d in range(nd))
        full_mask[original_slices] = True
        masks.append(full_mask)

    stacked = np.stack(padded, axis=0)
    mask_stack = np.stack(masks, axis=0)
    return stacked, mask_stack
