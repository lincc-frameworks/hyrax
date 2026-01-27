"""
Unit tests for LSST dataset classes using Butler mocks.

This test module demonstrates how to use the mock LSST Butler objects
to test LSSTDataset and DownloadedLSSTDataset without requiring actual
LSST Science Pipelines or a Butler repository.
"""

import threading
import unittest.mock as mock

import numpy as np
import pytest
from astropy.table import Table
from mocks.lsst_butler_mocks import MOCK_CUTOUT_SIZE, MOCK_IMAGE_MAX_SIZE, MockButler, MockGeom


@pytest.fixture
def mock_lsst_environment():
    """Fixture providing a complete mock LSST environment.

    This fixture patches the lsst.daf.butler and lsst.geom modules
    to use our mock implementations.
    """
    # Create mock modules
    mock_butler_module = mock.MagicMock()
    mock_butler_module.Butler = MockButler
    MockButler.initialized_thread_ids = []

    mock_geom_module = mock.MagicMock()
    mock_geom_module.Box2I = MockGeom.Box2I
    mock_geom_module.Box2D = MockGeom.Box2D
    mock_geom_module.SpherePoint = MockGeom.SpherePoint
    mock_geom_module.degrees = MockGeom.degrees(1.0)

    # Add modules to sys.modules before importing
    with mock.patch.dict(
        "sys.modules",
        {
            "lsst": mock.MagicMock(),
            "lsst.daf": mock.MagicMock(),
            "lsst.daf.butler": mock_butler_module,
            "lsst.geom": mock_geom_module,
        },
    ):
        yield {
            "butler": mock_butler_module,
            "geom": mock_geom_module,
        }


@pytest.fixture
def sample_catalog():
    """Create a sample astropy catalog for testing."""
    catalog_data = {
        "object_id": [1001, 1002, 1003, 1004, 1005],
        "coord_ra": [150.0, 150.1, 150.2, 150.3, 150.4],
        "coord_dec": [2.0, 2.1, 2.2, 2.3, 2.4],
    }
    return Table(catalog_data)


@pytest.fixture
def lsst_config():
    """Create a basic configuration for LSSTDataset."""
    return {
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
            "use_cache": True,
            "preload_cache": True,
        },
        "general": {
            "data_dir": "/tmp/test_data",
        },
    }


def test_mock_butler_basic_operations(mock_lsst_environment):
    """Test that mock Butler performs basic operations correctly."""
    # Create a mock butler
    butler = MockButler(repo="/fake/repo", collections="fake_collection")

    # Test getting a skymap
    skymap = butler.get("skyMap", {"skymap": "test_skymap"})
    assert skymap is not None
    skymap.reset()

    # Test finding a tract
    from mocks.lsst_butler_mocks import MockSpherePoint

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
    exposure = butler.get(
        "deep_coadd",
        {
            "tract": 9813,
            "patch": 42,
            "band": "g",
            "skymap": "test_skymap",
        },
    )
    assert exposure is not None

    # Test getting image from exposure
    image = exposure.getImage()
    assert image is not None

    # Test getting array from image
    arr = image.getArray()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (MOCK_IMAGE_MAX_SIZE, MOCK_IMAGE_MAX_SIZE)


def test_multiple_butler_per_thread_fails(mock_lsst_environment):
    """Test that mock butler can only be created once per thread"""
    # Create a mock butler
    _ = MockButler(repo="/fake/repo", collections="fake_collection")

    with pytest.raises(RuntimeError):
        # Create a second mock butler on the same thread
        MockButler(repo="/fake/repo", collections="fake_collection")


def test_one_butler_per_thread_succeeds(mock_lsst_environment):
    """Test that when several threads each make a single butler there are no crashes"""
    threads = [
        threading.Thread(target=MockButler, kwargs={"repo": "/fake/repo", "collections": "fake_collection"})
        for _ in range(5)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def test_mock_geom_operations(mock_lsst_environment):
    """Test that mock geometry operations work correctly."""
    geom_module = mock_lsst_environment["geom"]

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
    assert box2d.getMax() == [float(MOCK_CUTOUT_SIZE), float(MOCK_CUTOUT_SIZE)]

    # Test Box2I creation from Box2D
    box2i = geom_module.Box2I(box2d, geom_module.Box2I.EXPAND)
    assert box2i.getWidth() == MOCK_CUTOUT_SIZE
    assert box2i.getHeight() == MOCK_CUTOUT_SIZE
    assert not box2i.isEmpty()
