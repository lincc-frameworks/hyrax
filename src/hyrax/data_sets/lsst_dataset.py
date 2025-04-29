import functools

from torch.utils.data import Dataset

from .data_set_registry import HyraxDataset


class LSSTDataset(HyraxDataset, Dataset):
    """LSSTDataset: A dataset to access deep_coadd images from lsst pipelines
    via the butler. Must be run in an RSP.

    Input catalog is hardcoded
    Cutout code incorrect near the edges of a tract or patch

    Batteries not included, use at your own risk etc.
    """

    # BANDS = ["u", "g", "r", "i", "z", "y"]
    BANDS = ["g", "r", "i"]

    def __init__(self, config):
        try:
            import lsdb
            import lsst.daf.butler as butler
        except ImportError as e:
            msg = "LSSTDataset can only be used in a Rubin Software Platform environment"
            msg += " with access to the lsst pipeline tools and lsdb"
            raise ImportError(msg) from e

        self.butler = butler.Butler(
            config["data_set"]["butler_repo"], collections=config["data_set"]["butler_collection"]
        )
        self.skymap = self.butler.get("skyMap", {"skymap": config["data_set"]["skymap"]})

        # We compute the entire catalog so we have a nested frame which we can access.
        self.catalog = lsdb.read_hats(config["data_set"]["hats_catalog"]).compute()
        self.sh_deg = config["data_set"]["semi_height_deg"]
        self.sw_deg = config["data_set"]["semi_width_deg"]

        # TODO: Metadata from the catalog
        super().__init__(config)

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idxs):
        from nested_pandas import NestedFrame

        frame = self.catalog.iloc[idxs]
        frame = frame if isinstance(frame, NestedFrame) else NestedFrame(frame).T
        cutouts = [self._fetch_single_cutout(row) for _, row in frame.iterrows()]

        return cutouts if len(cutouts) > 1 else cutouts[0]

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
        # right-handed pixel space

        # Move + in ra (0.0) for width and - in dec (270.0) along a great circle
        min_pt_sky = radec.offset(0.0 * degrees, sw).offset(270.0 * degrees, sh)
        # Move - in ra (180.0) for width and + in dec (90.0) along a great circle
        max_pt_sky = radec.offset(180.0 * degrees, sw).offset(90.0 * degrees, sh)

        wcs = patch.getWcs()
        min_pt_pixel_f = wcs.skyToPixel(min_pt_sky)
        max_pt_pixel_f = wcs.skyToPixel(max_pt_sky)
        box_f = Box2D(min_pt_pixel_f, max_pt_pixel_f)
        return Box2I(box_f, Box2I.EXPAND)

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

        # Note: How to look up all tracts for the ra/dec (this will happen sometimes because of
        # overlapping tracts)
        #
        # [(tract, tract.findPatch(radec)) for tract in skymap.findAllTracts(radec)]
        # print(tractInfo.getId(), patchInfo.sequential_index)
        #
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

        return data_torch
