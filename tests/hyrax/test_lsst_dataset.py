"""
Unit tests for LSST dataset classes using Butler mocks.

This test module demonstrates how to use the mock LSST Butler objects
to test LSSTDataset and DownloadedLSSTDataset without requiring actual
LSST Science Pipelines or a Butler repository.
"""

import unittest.mock as mock

import pytest
import torch
from astropy.table import Table
from mocks.lsst_butler_mocks import mock_lsst_environment  # noqa: F401


@pytest.fixture(params=["fits", "hats"])
# @pytest.fixture(params=["fits"])
def sample_catalog(request, tmp_path):
    """Create a sample astropy catalog for testing."""
    catalog_data = {
        "object_id": [1001, 1002, 1003, 1004, 1005],
        "coord_ra": [150.0, 150.1, 150.2, 150.3, 150.4],
        "coord_dec": [2.0, 2.1, 2.2, 2.3, 2.4],
    }
    table = Table(catalog_data)

    print(request.param)
    catalog_type = request.param
    catalog_path = tmp_path / f"test_catalog.{catalog_type}"

    if catalog_type == "fits":
        table.write(catalog_path)
    elif catalog_type == "hats":
        import lsdb

        catalog = lsdb.from_dataframe(table.to_pandas(), ra_column="coord_ra", dec_column="coord_dec")
        catalog.to_hats(catalog_path)

    return catalog_type, catalog_path


@pytest.fixture
def lsst_config(sample_catalog):
    """Create a basic configuration for LSSTDataset."""
    config_dict = {
        "data_set": {
            "butler_repo": "/fake/butler/repo",
            "butler_collection": "fake_collection",
            "skymap": "fake_skymap",
            "semi_height_deg": 0.01,
            "semi_width_deg": 0.01,
            "object_id_column_name": "object_id",
            "filters": ["g", "r", "i"],
            "transform": "tanh",
            "crop_to": [100, 100],
            "use_cache": False,
            "preload_cache": False,
        },
        "general": {
            "data_dir": "/tmp/test_data",
        },
    }

    print(sample_catalog)

    if sample_catalog[0] == "fits":
        config_dict["data_set"]["astropy_table"] = sample_catalog[1]
    elif sample_catalog[0] == "hats":
        config_dict["data_set"]["hats_catalog"] = sample_catalog[1]

    print(config_dict)

    return config_dict


def test_lsst_dataset_init(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """Test LSSTDataset initialization and basic operations with mocks.

    This test demonstrates how to use the mocks to test the LSSTDataset class
    without requiring actual LSST infrastructure.
    """
    # Import after patching
    from hyrax.data_sets.lsst_dataset import LSSTDataset

    # Create LSSTDataset instance
    dataset = LSSTDataset(lsst_config, data_location=str(tmp_path))

    # Verify catalog was loaded
    assert dataset.catalog is not None
    assert len(dataset.catalog) == 5

    # Test basic dataset properties
    assert len(dataset) == 5

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
