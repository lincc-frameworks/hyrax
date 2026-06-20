from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from hyrax.verbs.visualize import Visualize


class FakeOriginalDataset:
    def __init__(self, paths_by_object):
        self.paths_by_object = paths_by_object

    def object_file_paths(self, object_id, filters=None):
        paths = self.paths_by_object[str(object_id)]
        if filters is None:
            return paths
        return {band: paths[band] for band in filters}


def test_validate_image_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="image_mode"):
        Visualize._validate_image_mode("rgb")


def test_load_fits_rgb_array_reads_requested_bands(tmp_path):
    paths_by_band = {}
    for band, value in [("I", 1.0), ("R", 2.0), ("G", 3.0)]:
        path = tmp_path / f"object_42_{band}.fits"
        fits.writeto(path, np.full((4, 5), value), overwrite=True)
        paths_by_band[band] = path

    viz = Visualize(config={})
    viz.umap_results = SimpleNamespace(
        original_dataset=FakeOriginalDataset({"42": paths_by_band})
    )
    viz.fits_rgb_bands = ["I", "R", "G"]

    rgb = viz._load_fits_rgb_array("42")

    assert rgb.shape == (4, 5, 3)
    np.testing.assert_array_equal(rgb[:, :, 0], np.full((4, 5), 1.0))
    np.testing.assert_array_equal(rgb[:, :, 1], np.full((4, 5), 2.0))
    np.testing.assert_array_equal(rgb[:, :, 2], np.full((4, 5), 3.0))
