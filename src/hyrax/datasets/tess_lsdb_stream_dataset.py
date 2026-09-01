import logging

import numpy as np

from hyrax.datasets.lsdb_stream_dataset import LSDBStreamDataset

logger = logging.getLogger(__name__)

MAX_ALLOWED_LENGTH = 50


class TessLSDBStreamDataset(LSDBStreamDataset):
    """This thin class in only meant to implement collation functions for two
    of the columns present in the nested TESS dataset.
    """

    def __init__(self, config, data_location):
        super().__init__(config, data_location)

    def collate_lightcurve_sap_flux(self, samples):
        """Collation function for Tess lightcurve sap flux."""
        result = {}

        field = "lightcurve_sap_flux"
        vals = [list(s[field]) for s in samples]
        lengths = np.array([len(s) for s in vals], dtype=np.int64)

        padded = np.zeros((len(vals), MAX_ALLOWED_LENGTH), dtype=np.float32)
        mask = np.zeros((len(vals), MAX_ALLOWED_LENGTH), dtype=np.int64)
        for i, s in enumerate(vals):
            length = min(lengths[i], MAX_ALLOWED_LENGTH)
            padded[i, :length] = s[:length]
            mask[i, :length] = 1

        result[field] = padded
        result[field + "_lengths"] = lengths
        result[field + "_mask"] = mask

        return result

    def collate_lightcurve_time(self, samples):
        """Collation function for Tess lightcurve time."""
        result = {}

        field = "lightcurve_time"
        vals = [list(s[field]) for s in samples]
        lengths = np.array([len(s) for s in vals], dtype=np.int64)

        padded = np.zeros((len(vals), MAX_ALLOWED_LENGTH), dtype=np.float32)
        mask = np.zeros((len(vals), MAX_ALLOWED_LENGTH), dtype=np.int64)
        for i, s in enumerate(vals):
            length = min(lengths[i], MAX_ALLOWED_LENGTH)
            padded[i, :length] = s[:length]
            mask[i, :length] = 1

        result[field] = padded
        result[field + "_lengths"] = lengths
        result[field + "_mask"] = mask

        return result
