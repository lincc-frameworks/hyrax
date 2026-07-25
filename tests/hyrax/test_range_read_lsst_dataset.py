"""Tests for RangeReadLSSTDataset using Butler mocks and real FITS files on disk."""

import unittest.mock as mock

import fitsio
import mocks
import numpy as np
import pytest
import torch
import torchvision  # noqa: F401
from mocks import lsst_config, mock_lsst_environment, sample_catalog, sample_catalog_saved  # noqa: F401
from mocks.lsst_butler_mocks import MOCK_IMAGE_MAX_SIZE, MockSkyMap


def _write_mock_fits_files(fits_dir, bands, tract_ids, patch_ids):
    """Write FITS files with deterministic data for testing.

    Each file has an empty primary HDU (HDU 0) and an image HDU (HDU 1) with
    shape (MOCK_IMAGE_MAX_SIZE, MOCK_IMAGE_MAX_SIZE) of float32 data.
    Matches the real LSST deep_coadd FITS structure.
    """
    from astropy.io.fits import HDUList, ImageHDU, PrimaryHDU

    for tract_id in tract_ids:
        for patch_id in patch_ids:
            for band in bands:
                seed = hash((tract_id, patch_id, band)) % (2**32)
                rng = np.random.RandomState(seed)
                image_data = rng.randn(MOCK_IMAGE_MAX_SIZE, MOCK_IMAGE_MAX_SIZE).astype(np.float32)

                path = fits_dir / f"deep_coadd_{tract_id}_{patch_id}_{band}.fits"
                hdul = HDUList([PrimaryHDU(), ImageHDU(data=image_data, name="IMAGE")])
                hdul.writeto(str(path), overwrite=True)


@pytest.fixture
def fits_dir_with_data(tmp_path):
    """Create a directory with mock FITS files matching the mock skymap's tract/patch IDs.

    MockTractInfo.findPatch always returns patch_id=42, so we only need files
    for each tract_id with patch_id=42.
    """
    fits_dir = tmp_path / "fits_data"
    fits_dir.mkdir()

    bands = ["g", "r", "i"]
    tract_ids = [info["tract_id"] for info in MockSkyMap.ids]
    _write_mock_fits_files(fits_dir, bands, tract_ids, [42])

    return fits_dir


def test_range_read_init_prefetch(mock_lsst_environment, lsst_config, fits_dir_with_data):  # noqa: F811
    """Test that RangeReadLSSTDataset prefetches file paths and row info during init."""
    with mock_lsst_environment(fits_dir=str(fits_dir_with_data)):
        from hyrax.datasets.range_read_lsst_dataset import RangeReadLSSTDataset

        dataset = RangeReadLSSTDataset(lsst_config)

        assert len(dataset._row_info) == mocks.SAMPLE_CATALOG_LENGTH
        assert len(dataset._file_paths) > 0

        for tract_id, patch_id, box_i, _origin_x, _origin_y in dataset._row_info:
            assert isinstance(tract_id, int)
            assert isinstance(patch_id, int)
            assert box_i.getWidth() > 0
            assert box_i.getHeight() > 0

        resolved = sum(1 for v in dataset._file_paths.values() if v is not None)
        assert resolved == len(dataset._file_paths)


def test_range_read_get_image_single(mock_lsst_environment, lsst_config, fits_dir_with_data):  # noqa: F811
    """Test that get_image returns a correctly shaped tensor for a single index."""
    with mock_lsst_environment(fits_dir=str(fits_dir_with_data)):
        from hyrax.datasets.range_read_lsst_dataset import RangeReadLSSTDataset

        dataset = RangeReadLSSTDataset(lsst_config)
        dataset.apply_transform = mock.MagicMock(side_effect=lambda x: x)

        cutout = dataset.get_image(0)

        assert isinstance(cutout, torch.Tensor)
        assert cutout.shape[0] == 3  # g, r, i bands

        # Verify shape matches the precomputed bounding box
        _, _, box_i, _, _ = dataset._row_info[0]
        assert cutout.shape[1] == box_i.getHeight()
        assert cutout.shape[2] == box_i.getWidth()
        assert not torch.any(torch.isnan(cutout))


def test_range_read_get_image_batch(mock_lsst_environment, lsst_config, fits_dir_with_data):  # noqa: F811
    """Test that get_image returns a list of tensors for multiple indices."""
    with mock_lsst_environment(fits_dir=str(fits_dir_with_data)):
        from hyrax.datasets.range_read_lsst_dataset import RangeReadLSSTDataset

        dataset = RangeReadLSSTDataset(lsst_config)
        dataset.apply_transform = mock.MagicMock(side_effect=lambda x: x)

        cutouts = dataset.get_image([0, 1, 2])

        assert isinstance(cutouts, list)
        assert len(cutouts) == 3
        for cutout in cutouts:
            assert isinstance(cutout, torch.Tensor)
            assert cutout.shape[0] == 3


def test_range_read_getitem(mock_lsst_environment, lsst_config, fits_dir_with_data):  # noqa: F811
    """Test that __getitem__ returns the expected dict structure."""
    with mock_lsst_environment(fits_dir=str(fits_dir_with_data)):
        from hyrax.datasets.range_read_lsst_dataset import RangeReadLSSTDataset

        dataset = RangeReadLSSTDataset(lsst_config)
        dataset.apply_transform = mock.MagicMock(side_effect=lambda x: x)

        result = dataset[0]

        assert "data" in result
        assert "image" in result["data"]
        assert isinstance(result["data"]["image"], torch.Tensor)


def test_range_read_missing_band(mock_lsst_environment, lsst_config, fits_dir_with_data):  # noqa: F811
    """Test that missing bands are filled with NaN."""
    with mock_lsst_environment(fits_dir=str(fits_dir_with_data), band_fail_prob={"g": 1.0}):
        from hyrax.datasets.range_read_lsst_dataset import RangeReadLSSTDataset

        dataset = RangeReadLSSTDataset(lsst_config)
        dataset.apply_transform = mock.MagicMock(side_effect=lambda x: x)

        cutout = dataset.get_image(0)

        assert isinstance(cutout, torch.Tensor)
        assert cutout.shape[0] == 3

        # Band 0 (g) should be all NaN since find_dataset returns None for it
        assert torch.all(torch.isnan(cutout[0]))
        # Bands 1,2 (r, i) should have real data
        assert not torch.any(torch.isnan(cutout[1]))
        assert not torch.any(torch.isnan(cutout[2]))


def test_range_read_missing_fits_file(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """Test that missing FITS files on disk result in NaN fill."""
    empty_fits_dir = tmp_path / "empty_fits"
    empty_fits_dir.mkdir()

    with mock_lsst_environment(fits_dir=str(empty_fits_dir)):
        from hyrax.datasets.range_read_lsst_dataset import RangeReadLSSTDataset

        dataset = RangeReadLSSTDataset(lsst_config)
        dataset.apply_transform = mock.MagicMock(side_effect=lambda x: x)

        cutout = dataset.get_image(0)

        assert isinstance(cutout, torch.Tensor)
        # All bands should be NaN since no FITS files exist
        assert torch.all(torch.isnan(cutout))


def test_range_read_no_butler_raises(mock_lsst_environment, lsst_config):  # noqa: F811
    """Test that RangeReadLSSTDataset raises when butler is not available."""
    # Don't use mock_lsst_environment — butler won't be available
    with pytest.raises(RuntimeError, match="requires a butler"):
        from hyrax.datasets.range_read_lsst_dataset import RangeReadLSSTDataset

        RangeReadLSSTDataset(lsst_config)


def test_range_read_correct_pixels(mock_lsst_environment, lsst_config, fits_dir_with_data):  # noqa: F811
    """Test that the range read returns the correct pixel values from the FITS file."""
    with mock_lsst_environment(fits_dir=str(fits_dir_with_data)):
        from hyrax.datasets.range_read_lsst_dataset import RangeReadLSSTDataset

        dataset = RangeReadLSSTDataset(lsst_config)
        dataset.apply_transform = mock.MagicMock(side_effect=lambda x: x)

        cutout = dataset.get_image(0)
        info = dataset._row_info[0]
        tract_id, patch_id, box_i, origin_x, origin_y = info

        # Read the same region directly with fitsio to verify
        band = dataset.BANDS[0]
        fits_path = fits_dir_with_data / f"deep_coadd_{tract_id}_{patch_id}_{band}.fits"

        y_start = box_i.getMinY() - origin_y
        y_end = y_start + box_i.getHeight()
        x_start = box_i.getMinX() - origin_x
        x_end = x_start + box_i.getWidth()

        with fitsio.FITS(str(fits_path), "r") as fits:
            expected = fits[1][y_start:y_end, x_start:x_end]

        np.testing.assert_array_equal(cutout[0].numpy(), expected)
