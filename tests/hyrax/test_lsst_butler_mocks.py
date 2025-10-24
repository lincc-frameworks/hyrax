"""
Unit tests for LSST dataset classes using Butler mocks.

This test module demonstrates how to use the mock LSST Butler objects
to test LSSTDataset and DownloadedLSSTDataset without requiring actual
LSST Science Pipelines or a Butler repository.
"""

import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
import torch
from astropy.table import Table

from tests.hyrax.mocks.lsst_butler_mocks import (
    MockButler,
    MockGeom,
    create_mock_butler_environment,
)


@pytest.fixture
def mock_lsst_environment():
    """Fixture providing a complete mock LSST environment.
    
    This fixture patches the lsst.daf.butler and lsst.geom modules
    to use our mock implementations.
    """
    # Create mock modules
    mock_butler_module = mock.MagicMock()
    mock_butler_module.Butler = MockButler
    
    mock_geom_module = mock.MagicMock()
    mock_geom_module.Box2I = MockGeom.Box2I
    mock_geom_module.Box2D = MockGeom.Box2D
    mock_geom_module.SpherePoint = MockGeom.SpherePoint
    mock_geom_module.degrees = MockGeom.degrees(1.0)
    
    # Add modules to sys.modules before importing
    with mock.patch.dict('sys.modules', {
        'lsst': mock.MagicMock(),
        'lsst.daf': mock.MagicMock(),
        'lsst.daf.butler': mock_butler_module,
        'lsst.geom': mock_geom_module,
    }):
        yield {
            'butler': mock_butler_module,
            'geom': mock_geom_module,
        }


@pytest.fixture
def sample_catalog():
    """Create a sample astropy catalog for testing."""
    catalog_data = {
        'object_id': [1001, 1002, 1003, 1004, 1005],
        'coord_ra': [150.0, 150.1, 150.2, 150.3, 150.4],
        'coord_dec': [2.0, 2.1, 2.2, 2.3, 2.4],
    }
    return Table(catalog_data)


@pytest.fixture
def lsst_config():
    """Create a basic configuration for LSSTDataset."""
    return {
        'data_set': {
            'butler_repo': '/fake/butler/repo',
            'butler_collection': 'fake_collection',
            'skymap': 'fake_skymap',
            'semi_height_deg': 0.01,
            'semi_width_deg': 0.01,
            'object_id_column_name': 'object_id',
            'filters': ['g', 'r', 'i'],
        },
        'general': {
            'data_dir': '/tmp/test_data',
        },
    }


def test_mock_butler_basic_operations(mock_lsst_environment):
    """Test that mock Butler performs basic operations correctly."""
    # Create a mock butler
    butler = MockButler(repo="/fake/repo", collections="fake_collection")
    
    # Test getting a skymap
    skymap = butler.get("skyMap", {"skymap": "test_skymap"})
    assert skymap is not None
    
    # Test finding a tract
    from tests.hyrax.mocks.lsst_butler_mocks import MockSpherePoint
    degrees = MockGeom.degrees(1.0)
    sphere_point = MockSpherePoint(150.0, 2.0, degrees)
    tract_info = skymap.findTract(sphere_point)
    
    assert tract_info is not None
    assert tract_info.getId() == 9813  # Mock returns fixed tract ID
    
    # Test finding a patch
    patch_info = tract_info.findPatch(sphere_point)
    assert patch_info is not None
    assert patch_info.sequential_index == 42  # Mock returns fixed patch index
    
    # Test getting an exposure
    exposure = butler.get("deep_coadd", {
        "tract": 9813,
        "patch": 42,
        "band": "g",
        "skymap": "test_skymap",
    })
    assert exposure is not None
    
    # Test getting image from exposure
    image = exposure.getImage()
    assert image is not None
    
    # Test getting array from image
    arr = image.getArray()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10000, 10000)


def test_mock_geom_operations(mock_lsst_environment):
    """Test that mock geometry operations work correctly."""
    geom_module = mock_lsst_environment['geom']
    
    # Test creating a SpherePoint
    degrees = geom_module.degrees
    sphere_point = geom_module.SpherePoint(150.0, 2.0, degrees)
    
    # Test getting coordinates
    ra = sphere_point.getLongitude().asDegrees()
    dec = sphere_point.getLatitude().asDegrees()
    assert ra == 150.0
    assert dec == 2.0
    
    # Test offset operation
    offset_point = sphere_point.offset(0.0 * degrees, 0.01 * degrees)
    assert offset_point is not None
    
    # Test Box2D creation
    box2d = geom_module.Box2D([0, 0], [100, 100])
    assert box2d.getMin() == [0.0, 0.0]
    assert box2d.getMax() == [100.0, 100.0]
    
    # Test Box2I creation from Box2D
    box2i = geom_module.Box2I(box2d, geom_module.Box2I.EXPAND)
    assert box2i.getWidth() == 100
    assert box2i.getHeight() == 100
    assert not box2i.isEmpty()


def test_lsst_dataset_with_mocks(mock_lsst_environment, sample_catalog, lsst_config, tmp_path):
    """Test LSSTDataset initialization and basic operations with mocks.
    
    This test demonstrates how to use the mocks to test the LSSTDataset class
    without requiring actual LSST infrastructure.
    """
    # Import after patching
    from hyrax.data_sets.lsst_dataset import LSSTDataset
    
    # Save catalog to a temporary file
    catalog_path = tmp_path / "test_catalog.fits"
    sample_catalog.write(catalog_path)
    
    # Update config to use the temporary catalog
    lsst_config['data_set']['astropy_table'] = str(catalog_path)
    
    # Mock the parent class __init__ to avoid issues
    with mock.patch('hyrax.data_sets.data_set_registry.HyraxDataset.__init__'):
        # Create LSSTDataset instance
        dataset = LSSTDataset(lsst_config, data_location=str(tmp_path))
        
        # Verify butler was created
        assert dataset.butler is not None
        assert dataset.skymap is not None
        
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


def test_downloaded_lsst_dataset_with_mocks(mock_lsst_environment, sample_catalog, lsst_config, tmp_path):
    """Test DownloadedLSSTDataset with mocks.
    
    This test demonstrates testing the downloaded dataset variant which
    caches cutouts to disk.
    """
    # Import after patching
    from hyrax.data_sets.downloaded_lsst_dataset import DownloadedLSSTDataset
    
    # Save catalog to a temporary file
    catalog_path = tmp_path / "test_catalog.fits"
    sample_catalog.write(catalog_path)
    
    # Update config
    lsst_config['data_set']['astropy_table'] = str(catalog_path)
    lsst_config['general']['data_dir'] = str(tmp_path)
    
    # Mock the parent class init and problematic methods
    with mock.patch('hyrax.data_sets.lsst_dataset.LSSTDataset.__init__'):
        with mock.patch.object(DownloadedLSSTDataset, '_initialize_manifest'):
            # Create DownloadedLSSTDataset instance
            dataset = DownloadedLSSTDataset(lsst_config, data_location=str(tmp_path))
            
            # Set required attributes manually for testing
            dataset.catalog = sample_catalog
            dataset.BANDS = ('g', 'r', 'i')
            dataset.config = lsst_config
            dataset.download_dir = tmp_path
            dataset._object_id_column_name = 'object_id'
            dataset.butler = MockButler(
                repo=lsst_config['data_set']['butler_repo'],
                collections=lsst_config['data_set']['butler_collection']
            )
            dataset.skymap = dataset.butler.get("skyMap", {"skymap": lsst_config['data_set']['skymap']})
            dataset.sh_deg = lsst_config['data_set']['semi_height_deg']
            dataset.sw_deg = lsst_config['data_set']['semi_width_deg']
            
            # Mock transform methods
            dataset.apply_transform = mock.MagicMock(side_effect=lambda x: x)
            
            # Create a minimal manifest
            manifest_data = {
                'object_id': sample_catalog['object_id'],
                'cutout_shape': [np.array([3, 100, 100], dtype=int) for _ in range(len(sample_catalog))],
                'filename': [f"cutout_{oid}.pt" for oid in sample_catalog['object_id']],
                'downloaded_bands': ['g,r,i' for _ in range(len(sample_catalog))],
            }
            dataset.manifest = Table(manifest_data)
            dataset._is_filtering_bands = False
            dataset._band_indices = None
            dataset._original_bands = ('g', 'r', 'i')
            dataset._catalog_to_manifest_index_map = None
            dataset._manifest_to_catalog_index_map = None
            
            # Test that dataset has correct length
            assert len(dataset) == 5
            
            # Test _fetch_cutout_with_cache method
            row = dataset.catalog[0]
            cutout, downloaded_bands = dataset._fetch_cutout_with_cache(row)
            
            # Verify cutout is a tensor
            assert isinstance(cutout, torch.Tensor)
            
            # Verify it has the right number of bands
            assert cutout.shape[0] == 3
            
            # Verify downloaded_bands is correct
            assert downloaded_bands == ['g', 'r', 'i']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
