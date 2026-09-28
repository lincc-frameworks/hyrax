"""
Unit tests for LSST dataset classes using Butler mocks.

This test module demonstrates how to use the mock LSST Butler objects
to test LSSTDataset and DownloadedLSSTDataset without requiring actual
LSST Science Pipelines or a Butler repository.
"""

import unittest.mock as mock

import mocks
import torch
import torchvision  # noqa: F401  # Import before mock contexts to prevent kernel re-registration
from mocks import lsst_config, mock_lsst_environment, sample_catalog, sample_catalog_saved  # noqa: F401
from mocks.lsst_butler_mocks import MockButler


def test_lsst_dataset_init(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """Test LSSTDataset initialization and basic operations with mocks.

    This test demonstrates how to use the mocks to test the LSSTDataset class
    without requiring actual LSST infrastructure.
    """
    with mock_lsst_environment():
        # Import after patching
        from hyrax.datasets.lsst_dataset import LSSTDataset

        # Create LSSTDataset instance
        dataset = LSSTDataset(lsst_config, data_location=str(tmp_path))

        # Verify catalog was loaded
        assert dataset.catalog is not None
        assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

        # Test basic dataset properties
        assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

        # Mock the transform methods to avoid issues
        dataset.set_function_transform = mock.MagicMock()
        dataset.set_crop_transform = mock.MagicMock()
        dataset.apply_transform = mock.MagicMock(side_effect=lambda x: x)

        # Test fetching a single cutout
        row = dataset.catalog[0]

        # The _fetch_single_cutout method should work with mocks
        cutout = dataset._fetch_single_cutout(row)

        # Verify cutout is a tensor
        assert isinstance(cutout, torch.Tensor)

        # Verify it has the right number of bands (channels)
        assert cutout.shape[0] == 3  # g, r, i bands


def test_lsst_dataset_band_failures(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """
    Test LSSTDataset can handle a download where some butler gets are failing, and
    fills missing bands with NaN
    """

    with mock_lsst_environment(band_fail_prob={"g": 1.0}):
        # Import after patching
        from hyrax.datasets.lsst_dataset import LSSTDataset

        # Create LSSTDataset instance
        dataset = LSSTDataset(lsst_config, data_location=str(tmp_path))

        # Verify catalog was loaded
        assert dataset.catalog is not None
        assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

        # Test basic dataset properties
        assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

        # Mock the transform methods to avoid issues
        dataset.set_function_transform = mock.MagicMock()
        dataset.set_crop_transform = mock.MagicMock()
        dataset.apply_transform = mock.MagicMock(side_effect=lambda x: x)

        # Test fetching every cutout to see it doesn't crash
        # Look for nans in a particular band
        all_nans_band = False
        all_numbers_band = False
        for i in range(mocks.SAMPLE_CATALOG_LENGTH):
            cutout = dataset.get_image(i)
            for band in cutout:
                if torch.all(torch.isnan(band)):
                    all_nans_band = True
                if not torch.any(torch.isnan(band)):
                    all_numbers_band = True

        # Test that we have filled some bands with NaN
        assert all_nans_band
        assert all_numbers_band


def test_lsst_dataset_requests_exposure_storage_class(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """Test LSSTDataset requests deep_coadd data as Exposure."""
    with mock_lsst_environment():
        from hyrax.datasets.lsst_dataset import LSSTDataset

        dataset = LSSTDataset(lsst_config, data_location=str(tmp_path))
        dataset.get_image(0)

        deep_coadd_calls = [call for call in MockButler.get_calls if call["dataset_type"] == "deep_coadd"]
        assert deep_coadd_calls
        assert all(call["kwargs"].get("storageClass") == "Exposure" for call in deep_coadd_calls)
