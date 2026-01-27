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


def test_lsst_dataset_init(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """Test LSSTDataset initialization and basic operations with mocks.

    This test demonstrates how to use the mocks to test the LSSTDataset class
    without requiring actual LSST infrastructure.
    """
    with mock_lsst_environment():
        # Import after patching
        from hyrax.data_sets.lsst_dataset import LSSTDataset

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
