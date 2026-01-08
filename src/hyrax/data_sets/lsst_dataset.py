import functools
import logging
import threading
from pathlib import Path

from torch.utils.data import Dataset

from .data_set_registry import HyraxDataset, HyraxImageDataset

logger = logging.getLogger(__name__)


class LSSTDataset(HyraxDataset, HyraxImageDataset, Dataset):
    """LSSTDataset: A dataset to access deep_coadd images from lsst pipelines
    via the butler. Must be run in an RSP.
    """

    BANDS = ["u", "g", "r", "i", "z", "y"]
    object_id_autodetect_names = ["object_id", "objectId"]

    def __init__(self, config, data_location=None):
        """
        .. py:method:: __init__

        Initialize the dataset with either a HATS catalog or astropy table.

        Config can specify either:
        - config["data_set"]["hats_catalog"]: path to HATS catalog
        - config["data_set"]["astropy_table"]: path to any file readable by Astropy Table

        """

        if self._butler_available():
            self._butler_config = {
                "repo": config["data_set"]["butler_repo"],
                "collections": config["data_set"]["butler_collection"],
                "skymap": config["data_set"]["skymap"],
            }

            self._threaded_butler = {}
            self._threaded_butler_update_lock = threading.Lock()
        else:
            msg = "Did not detect a Butler. You may need to run on the RSP"
            msg += ""
            logger.info(msg)
            self._butler_config = None

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

        self.oid_column_name = (
            config["data_set"]["object_id_column_name"]
            if config["data_set"]["object_id_column_name"]
            else self._detect_object_id_column_name()
        )

        # TODO: Metadata from the catalog
        super().__init__(config, self.catalog, self.oid_column_name)

        self.set_function_transform()
        self.set_crop_transform()

    def _butler_available(self):
        try:
            import lsst.daf.butler as _butler  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_butler_thread_safe(self):
        """Thread safe butler creation

        This function ensures that there is one and only one butler created per thread
        and that threads always use their assigned butler.

        This is necessary because child classes of this one use butlers, and butler
        objects are not safe for multithreaded access.

        Returns
        -------
        butler
            The butler assigned to the current thread.
        """
        import lsst.daf.butler as butler

        thread_ident = threading.current_thread().ident

        # Try to get the correct butler for this thread.
        our_butler = self._threaded_butler.get(thread_ident, None)

        # If we can't get the right butler, grab the lock
        # (ensuring nobody else is creating one) and make our butler
        # This process relies on thread idents being unique, and there only being one
        # LSSTDataset or derived class in runtime at once.
        if our_butler is None:
            with self._threaded_butler_update_lock:
                repo = self._butler_config["repo"]
                collections = self._butler_config["collections"]
                our_butler = butler.Butler(repo, collections=collections)
                self._threaded_butler[thread_ident] = our_butler

        return our_butler

    def _detect_object_id_column_name(self):
        """Setup file naming strategy based on catalog columns."""
        catalog_columns = self.catalog.colnames if hasattr(self.catalog, "colnames") else self.catalog.columns

        # Autodetect ID column
        for object_id_name in LSSTDataset.object_id_autodetect_names:
            if object_id_name in catalog_columns:
                object_id_column_name = object_id_name
                break
        else:
            msg = "Must provide an Object ID column in your catalog. This ID must be unique and is used to\n"
            msg += "track in-progress downloads. It need not be a Rubin generated objectId, but could be.\n"
            msg += "You can configure the name of your object ID column with \n"
            msg += "config['data_set']['object_id_column_name']. \n"
            msg += "If nothing is configured, a column named 'object_id' or 'objectId' will be used\n"
            msg += "automatically if present in your catalog.\n"
            raise RuntimeError(msg)

        return object_id_column_name

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
        from astropy.table import Table

        try:
            import lsdb
        except ImportError as e:
            msg = "LSDB is required to load HATS catalogs. Install with: pip install lsdb"
            raise ImportError(msg) from e

        # We compute the entire catalog so we have a nested frame which we can access
        return Table.from_pandas(lsdb.read_hats(hats_path).compute())

    def _load_astropy_catalog(self, table_path):
        """Load catalog from astropy table format or pickled astropy table."""

        import pickle

        from astropy.table import Table

        table_path = Path(table_path)

        # Check if it's a pickle file
        # Loading a pickle file is significantly faster than Table.read()
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

        # Astropy table - extract rows directly
        if isinstance(idxs, (list, tuple)):
            rows = [self.catalog[idx] for idx in idxs]
            cutouts = [self._fetch_single_cutout(row) for row in rows]
            return cutouts
        else:
            row = self.catalog[idxs]
            return self._fetch_single_cutout(row)

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
        skymap = self._get_butler_thread_safe().get("skyMap", {"skymap": self._butler_config["skymap"]})
        tract_info = skymap.findTract(radec)
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
            image = self._get_butler_thread_safe().get("deep_coadd", butler_dict)
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
