import functools
import logging
from pathlib import Path

from torch.utils.data import Dataset

from .data_set_registry import HyraxDataset, HyraxImageDataset

logger = logging.getLogger(__name__)


class LSSTDataset(HyraxDataset, HyraxImageDataset, Dataset):
    """LSSTDataset: A dataset to access deep_coadd images from lsst pipelines
    via the butler. Must be run in an RSP.
    """

    BANDS = ["u", "g", "r", "i", "z", "y"]

    def __init__(self, config, data_location):
        """
        .. py:method:: __init__

        Initialize the dataset with either a HATS catalog or astropy table.

        Config can specify either:
        - config["data_set"]["hats_catalog"]: path to HATS catalog
        - config["data_set"]["astropy_table"]: path to any file readable by Astropy Table

        """
        try:
            import lsst.daf.butler as butler

            self.butler = butler.Butler(
                config["data_set"]["butler_repo"], collections=config["data_set"]["butler_collection"]
            )
            self.skymap = self.butler.get("skyMap", {"skymap": config["data_set"]["skymap"]})
        except ImportError:
            msg = "Did not detect a Butler. You may need to run on the RSP"
            msg += ""
            logger.info(msg)
            self.butler = None
            self.skymap = None

        # Set filters from config if provided, otherwise use class default
        if "filters" in config["data_set"] and config["data_set"]["filters"]:
            # Validate filters
            valid_filters = ["u", "g", "r", "i", "z", "y"]
            for band in config["data_set"]["filters"]:
                if band not in valid_filters:
                    raise ValueError(
                        f"Invalid filter {band} for Rubin-LSST.\
                                        Valid bands are: {valid_filters}"
                    )

            LSSTDataset.BANDS = config["data_set"]["filters"]

        # Load catalog - either from HATS or astropy table
        self.catalog = self._load_catalog(config["data_set"])

        self.sh_deg = config["data_set"]["semi_height_deg"]
        self.sw_deg = config["data_set"]["semi_width_deg"]

        self.set_function_transform()
        self.set_crop_transform()

        # TODO: Metadata from the catalog
        super().__init__(config)

    def _load_catalog(self, data_set_config):
        """
        Load the catalog from either a HATS catalog or an astropy table.
        """
        if "hats_catalog" in data_set_config:
            return self._load_hats_catalog(data_set_config["hats_catalog"])
        elif "astropy_table" in data_set_config:
            return self._load_astropy_catalog(data_set_config["astropy_table"])
        else:
            raise ValueError("Must specify either 'hats_catalog' or 'astropy_table' in data_set config")

    def _load_hats_catalog(self, hats_path):
        """Load catalog from HATS format using LSDB."""
        try:
            import lsdb
        except ImportError as e:
            msg = "LSDB is required to load HATS catalogs. Install with: pip install lsdb"
            raise ImportError(msg) from e

        # We compute the entire catalog so we have a nested frame which we can access
        return lsdb.read_hats(hats_path).compute()

    def _load_astropy_catalog(self, table_path):
        """Load catalog from astropy table format or pickled astropy table."""

        import pickle

        from astropy.table import Table

        table_path = Path(table_path)

        # Check if it's a pickle file
        if table_path.suffix.lower() in [".pkl", ".pickle"]:
            with open(table_path, "rb") as f:
                table = pickle.load(f)
            # Verify it's an astropy Table
            if not isinstance(table, Table):
                raise ValueError(f"Pickled file {table_path} does not contain an astropy Table")
            return table
        else:
            # Load using astropy's native readers -- can be any format supported by astropy
            return Table.read(table_path)

    def __len__(self):
        return len(self.catalog)

    def get_image(self, idxs):
        """Get image cutouts for the given indices.

        Parameters
        ----------
        idxs : int or list of int
            The index or indices of the cutouts to retrieve.

        Returns
        -------
        list or torch.Tensor
            Single cutout tensor or list of cutout tensors.
        """

        from astropy.table import Table
        from nested_pandas import NestedFrame

        # Handle different catalog types
        if isinstance(self.catalog, Table):
            # Astropy table - extract rows directly
            if isinstance(idxs, (list, tuple)):
                rows = [self.catalog[idx] for idx in idxs]
                cutouts = [self._fetch_single_cutout(row) for row in rows]
                return cutouts
            else:
                row = self.catalog[idxs]
                return self._fetch_single_cutout(row)
        else:
            # NestedFrame (HATS catalog)
            frame = self.catalog.iloc[idxs]
            frame = frame if isinstance(frame, NestedFrame) else NestedFrame(frame).T
            cutouts = [self._fetch_single_cutout(row) for _, row in frame.iterrows()]
            return cutouts if len(cutouts) > 1 else cutouts[0]

    def __getitem__(self, idxs):
        """Get default data fields for the this dataset.

        Parameters
        ----------
        idxs : int or list of int
            The index or indices of the cutouts to retrieve.

        Returns
        -------
        dict
            A dictionary containing the default data fields.
        """

        return {"data": {"image": self.get_image(idxs)}}

    # def __getitems__(self, idxs):
    #     return __getitem__(self, idxs)

    def _parse_box(self, patch, row):
        """
        Return a Box2I representing the desired cutout in pixel space, given a "row" of catalog data
        which includes the semi-height (sh) and semi-width (sw) in degrees desired for the cutout.
        """
        from lsst.geom import Box2D, Box2I, degrees

        radec = self._parse_sphere_point(row)
        sw = self.sh_deg * degrees
        sh = self.sw_deg * degrees

        # Ra/Dec is left handed on the sky. Pixel coordinates are right handed on the sky.
        # In the variable names below min/max mean the min/max coordinate values in the
        # right-handed pixel space, not the left-handed sky space.

        # Move + in ra (0.0) for width and - in dec (270.0) along a great circle
        min_pt_sky = radec.offset(0.0 * degrees, sw).offset(270.0 * degrees, sh)
        # Move - in ra (180.0) for width and + in dec (90.0) along a great circle
        max_pt_sky = radec.offset(180.0 * degrees, sw).offset(90.0 * degrees, sh)

        wcs = patch.getWcs()
        minmax_pt_pixel_f = wcs.skyToPixel([min_pt_sky, max_pt_sky])
        box_pixel_f = Box2D(*minmax_pt_pixel_f)
        box_pixel_i = Box2I(box_pixel_f, Box2I.EXPAND)

        # Throw if box_pixel_i extends outside the patch outer bbox
        # TODO: Do we want to fill nan's in this case?
        #       Do we want to conditionally fill nan's if a nan-infill strategy is configured in
        #       hyrax and error otherwise?
        if not patch.getOuterBBox().contains(box_pixel_i):
            msg = f"Bounding box for object at ra {radec.getLongitude().asDegrees()} deg "
            msg += f"dec {radec.getLatitude().asDegrees()} with semi-height {sh.asArcseconds()} arcsec "
            msg += f"and semi-width {sh.asArcseconds()} arcsec extends outside the bounding box of a "
            msg += "patch. Choose smaller values for config['data_set']['semi_height_deg'] and "
            msg += "config['data_set']['semi_width_deg']."
            raise RuntimeError(msg)

        # Throw if box_pixel_i does not contain any points
        if box_pixel_i.isEmpty():
            msg = "Calculated size for cutout is 0x0 pixels. Did you set "
            msg += "config['data_set']['semi_height_deg'] and config['data_set']['semi_width_deg']?"
            raise RuntimeError(msg)

        return box_pixel_i

    def _parse_sphere_point(self, row):
        """
        Return a SpherePoint with the ra and deck given in the "row" of catalog data.
        Row must include the RA and dec as "ra" and "dec" columns respectively
        """
        from lsst.geom import SpherePoint, degrees

        ra = row["coord_ra"]
        dec = row["coord_dec"]
        return SpherePoint(ra, dec, degrees)

    def _get_tract_patch(self, row):
        """
        Return (tractInfo, patchInfo) for a given row.

        This function only returns the single principle tract and patch in the case of overlap.
        """
        radec = self._parse_sphere_point(row)
        tract_info = self.skymap.findTract(radec)
        return (tract_info, tract_info.findPatch(radec))

    # super basic patch caching
    @functools.lru_cache(maxsize=128)  # noqa: B019
    def _request_patch(self, tract_index, patch_index):
        """
        Request a patch from the butler. This will be a list of
        lsst.afw.image objects each corresponding to the configured
        bands

        Uses functools.lru_cache for basic in-memory caching.
        """
        data = []

        # Get the patch images we need
        for band in LSSTDataset.BANDS:
            # Set up the data dict
            butler_dict = {
                "tract": tract_index,
                "patch": patch_index,
                "skymap": self.config["data_set"]["skymap"],
                "band": band,
            }

            # pull from butler
            image = self.butler.get("deep_coadd", butler_dict)
            data.append(image.getImage())
        return data

    def _fetch_single_cutout(self, row):
        """
        Make a single cutout, returning a torch tensor.

        Does not handle edge-of-tract/patch type edge cases, will only work near
        center of a patch.
        """
        import numpy as np
        from torch import from_numpy

        tract_info, patch_info = self._get_tract_patch(row)
        box_i = self._parse_box(patch_info, row)

        patch_images = self._request_patch(tract_info.getId(), patch_info.sequential_index)

        # Actually perform a cutout
        data = [image[box_i].getArray() for image in patch_images]

        # Convert to torch format
        data_np = np.array(data)
        data_torch = from_numpy(data_np.astype(np.float32))

        return self.apply_transform(data_torch)
