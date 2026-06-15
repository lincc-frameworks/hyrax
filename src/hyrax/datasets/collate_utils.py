import numpy as np


def collate_as_1d_light_curve(samples: list[dict], field: str) -> dict:
    """Collate the given field in the samples as if it were a light curve

    Parameters
    ----------
    samples
        List of dicts; each dict is expected to have the
        key passed in for the `field` argument
    field
        The field to collate

    Returns
    --------
    dict
        Contains three keys: `<field>`, `<field>_length`, and `<field>_mask`
        `field` - float32 array (batch, max_len) containing the padded light curves
        `<field>_length` - int64 array (batch) of true light curve lengths
        `<field>_mask` - int64 array (batch, max_len) of masks denoting light-curve data vs. padding
    """

    result = {}

    vals = [s[field] for s in samples]
    lengths = np.array([len(s) for s in vals], dtype=np.int64)
    max_len = int(lengths.max())

    padded = np.zeros((len(vals), max_len), dtype=np.float32)
    mask = np.zeros((len(vals), max_len), dtype=np.int64)
    for i, s in enumerate(vals):
        padded[i, : lengths[i]] = s
        mask[i, : lengths[i]] = 1

    result[field] = padded
    result[field + "_lengths"] = lengths
    result[field + "_mask"] = mask

    return result
