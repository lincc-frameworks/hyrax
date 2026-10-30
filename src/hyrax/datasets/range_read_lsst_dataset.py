import logging

import numpy as np
from torch import from_numpy

from .lsst_dataset import LSSTDataset

logger = logging.getLogger(__name__)


class RangeReadLSSTDataset(LSSTDataset):
    """LSST dataset that prefetches file paths from butler and reads cutouts via FITS range reads."""

    def __init__(self, config, data_location=None):
        super().__init__(config, data_location=data_location)

        if self._butler_config is None:
            raise RuntimeError(
                "RangeReadLSSTDataset requires a butler to resolve file paths during initialization. "
                "Ensure you are running on an environment with butler access."
            )

        self._prefetch_all()

    def _prefetch_all(self):
        """Resolve all file paths and pixel bounding boxes from the butler and skymap.

        After this method completes, the butler is no longer needed for get_image calls.
        """
        butler = self._get_butler_thread_safe()
        skymap = butler.get("skyMap", {"skymap": self._butler_config["skymap"]})

        self._row_info = []
        self._file_paths = {}

        for i in range(len(self.catalog)):
            row = self.catalog[i]
            radec = self._parse_sphere_point(row)
            tract_info = skymap.findTract(radec)
            patch_info = tract_info.findPatch(radec)

            box_i = self._parse_box(patch_info, row)
            origin = patch_info.getOuterBBox().getMin()

            tract_id = tract_info.getId()
            patch_id = patch_info.sequential_index

            self._row_info.append((tract_id, patch_id, box_i, origin.getX(), origin.getY()))

            for band in self.BANDS:
                key = (tract_id, patch_id, band)
                if key in self._file_paths:
                    continue

                try:
                    data_id = {
                        "tract": tract_id,
                        "patch": patch_id,
                        "skymap": self._butler_config["skymap"],
                        "band": band,
                    }
                    ref = butler.find_dataset("deep_coadd", data_id)
                    if ref is None:
                        logger.warning(f"No dataset found for {key}")
                        self._file_paths[key] = None
                        continue
                    uri = butler.getURI(ref)
                    self._file_paths[key] = uri.ospath
                except Exception as e:
                    logger.warning(f"Failed to resolve URI for {key}: {e}")
                    self._file_paths[key] = None

        resolved = sum(1 for v in self._file_paths.values() if v is not None)
        logger.info(
            f"Prefetch complete: {len(self._row_info)} rows, "
            f"{resolved}/{len(self._file_paths)} file paths resolved"
        )

    def get_image(self, idxs):
        """Get image cutouts for the given indices using FITS range reads.

        Parameters
        ----------
        idxs : int or list of int
            The index or indices of the cutouts to retrieve.

        Returns
        -------
        list or torch.Tensor
            Single cutout tensor or list of cutout tensors.
        """
        if isinstance(idxs, (list, tuple)):
            return [self._fetch_single_cutout_range_read(idx) for idx in idxs]
        return self._fetch_single_cutout_range_read(idxs)

    def _fetch_single_cutout_range_read(self, idx):
        """Read a single cutout from FITS files using range reads."""
        import fitsio

        tract_id, patch_id, box_i, origin_x, origin_y = self._row_info[idx]

        y_start = box_i.getMinY() - origin_y
        y_end = y_start + box_i.getHeight()
        x_start = box_i.getMinX() - origin_x
        x_end = x_start + box_i.getWidth()

        cutout_data = []
        for band in self.BANDS:
            file_path = self._file_paths.get((tract_id, patch_id, band))

            if file_path is None:
                nan_array = np.full((box_i.getHeight(), box_i.getWidth()), np.nan, dtype=np.float32)
                cutout_data.append(nan_array)
                logger.debug(f"No file path for band {band} at patch {tract_id}-{patch_id}, filling NaN")
                continue

            try:
                with fitsio.FITS(file_path, "r") as fits:
                    cutout_data.append(fits[1][y_start:y_end, x_start:x_end])
            except Exception as e:
                nan_array = np.full((box_i.getHeight(), box_i.getWidth()), np.nan, dtype=np.float32)
                cutout_data.append(nan_array)
                logger.debug(f"Failed to read band {band} from {file_path}: {e}")

        data_np = np.array(cutout_data)
        data_torch = from_numpy(data_np.astype(np.float32))
        return self.apply_transform(data_torch)
