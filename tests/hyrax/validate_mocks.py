"""
Standalone validation script for LSST Butler mocks.

This script tests the mock objects independently of the full hyrax package
to ensure they work correctly.
"""

import sys
from pathlib import Path

# Add the tests directory to the path
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir))

import numpy as np

from mocks.lsst_butler_mocks import (
    MockButler,
    MockGeom,
    MockSpherePoint,
    create_mock_butler_environment,
)


def test_basic_butler_operations():
    """Test basic butler operations."""
    print("Testing basic Butler operations...")
    
    # Create mock butler
    butler = MockButler(repo="/fake/repo", collections="fake_collection")
    print(f"✓ Created MockButler: {butler}")
    
    # Get skymap
    skymap = butler.get("skyMap", {"skymap": "test_skymap"})
    print(f"✓ Retrieved skymap: {skymap}")
    
    # Create a sphere point
    degrees = MockGeom.degrees(1.0)
    sphere_point = MockSpherePoint(150.0, 2.0, degrees)
    print(f"✓ Created SpherePoint at RA={sphere_point._ra}, Dec={sphere_point._dec}")
    
    # Find tract
    tract_info = skymap.findTract(sphere_point)
    print(f"✓ Found tract: {tract_info.getId()}")
    
    # Find patch
    patch_info = tract_info.findPatch(sphere_point)
    print(f"✓ Found patch: {patch_info.sequential_index}")
    
    # Get exposure
    exposure = butler.get("deep_coadd", {
        "tract": tract_info.getId(),
        "patch": patch_info.sequential_index,
        "band": "g",
        "skymap": "test_skymap",
    })
    print(f"✓ Retrieved exposure: {exposure}")
    
    # Get image
    image = exposure.getImage()
    print(f"✓ Retrieved image: {image}")
    
    # Get array
    arr = image.getArray()
    print(f"✓ Retrieved array with shape: {arr.shape}")
    
    assert arr.shape == (10000, 10000), f"Expected shape (10000, 10000), got {arr.shape}"
    assert isinstance(arr, np.ndarray), f"Expected numpy array, got {type(arr)}"
    
    print("✓ All basic Butler operations passed!")
    return True


def test_geometry_operations():
    """Test geometry operations."""
    print("\nTesting geometry operations...")
    
    # Test degrees
    degrees = MockGeom.degrees(1.0)
    print(f"✓ Created degrees object: {degrees.asDegrees()}")
    
    # Test multiplication
    scaled_deg = 0.01 * degrees
    print(f"✓ Scaled degrees: {scaled_deg.asDegrees()}")
    
    # Test SpherePoint
    sp = MockSpherePoint(150.0, 2.0, degrees)
    ra = sp.getLongitude().asDegrees()
    dec = sp.getLatitude().asDegrees()
    print(f"✓ SpherePoint coordinates: RA={ra}, Dec={dec}")
    
    # Test offset
    offset_sp = sp.offset(0.0 * degrees, 0.01 * degrees)
    new_ra = offset_sp.getLongitude().asDegrees()
    print(f"✓ Offset SpherePoint: RA={new_ra}")
    
    # Test Box2D
    box2d = MockGeom.Box2D([0, 0], [100, 100])
    print(f"✓ Created Box2D: min={box2d.getMin()}, max={box2d.getMax()}")
    
    # Test Box2I from Box2D
    box2i = MockGeom.Box2I(box2d, MockGeom.Box2I.EXPAND)
    print(f"✓ Created Box2I: width={box2i.getWidth()}, height={box2i.getHeight()}")
    
    assert box2i.getWidth() == 100, f"Expected width 100, got {box2i.getWidth()}"
    assert box2i.getHeight() == 100, f"Expected height 100, got {box2i.getHeight()}"
    assert not box2i.isEmpty(), "Box should not be empty"
    
    print("✓ All geometry operations passed!")
    return True


def test_image_cutout_operations():
    """Test image cutout operations."""
    print("\nTesting image cutout operations...")
    
    # Create a large image
    from mocks.lsst_butler_mocks import MockImage, MockBox2I
    
    large_image = MockImage(shape=(1000, 1000))
    arr = large_image.getArray()
    print(f"✓ Created large image with shape: {arr.shape}")
    
    # Create a box for cutout
    box = MockBox2I()
    box._min_x = 100
    box._min_y = 100
    box._max_x = 200
    box._max_y = 200
    print(f"✓ Created cutout box: {box}")
    
    # Extract cutout
    cutout_image = large_image[box]
    cutout_arr = cutout_image.getArray()
    print(f"✓ Extracted cutout with shape: {cutout_arr.shape}")
    
    expected_shape = (100, 100)  # 200-100 = 100 for both dimensions
    assert cutout_arr.shape == expected_shape, f"Expected shape {expected_shape}, got {cutout_arr.shape}"
    
    print("✓ All image cutout operations passed!")
    return True


def test_wcs_transformations():
    """Test WCS coordinate transformations."""
    print("\nTesting WCS transformations...")
    
    from mocks.lsst_butler_mocks import MockWcs, MockSpherePoint
    
    wcs = MockWcs()
    degrees = MockGeom.degrees(1.0)
    
    # Create some sky points
    sp1 = MockSpherePoint(150.0, 2.0, degrees)
    sp2 = MockSpherePoint(150.1, 2.1, degrees)
    
    # Transform to pixels
    pixel_coords = wcs.skyToPixel([sp1, sp2])
    print(f"✓ Transformed 2 sky points to pixels: {pixel_coords}")
    
    assert len(pixel_coords) == 2, f"Expected 2 pixel coordinates, got {len(pixel_coords)}"
    assert len(pixel_coords[0]) == 2, "Each pixel coordinate should have 2 values (x, y)"
    
    print("✓ All WCS transformations passed!")
    return True


def test_multi_band_retrieval():
    """Test retrieving images in multiple bands."""
    print("\nTesting multi-band image retrieval...")
    
    butler = MockButler()
    
    bands = ['g', 'r', 'i', 'z', 'y']
    images = []
    
    for band in bands:
        exposure = butler.get("deep_coadd", {
            "tract": 9813,
            "patch": 42,
            "band": band,
            "skymap": "test_skymap",
        })
        image = exposure.getImage()
        images.append(image)
        print(f"✓ Retrieved image for band {band}")
    
    assert len(images) == 5, f"Expected 5 images, got {len(images)}"
    
    # Verify each band produces different data (due to seeding with band name)
    arrays = [img.getArray() for img in images]
    
    # Check that not all arrays are identical (they should differ due to different seeds)
    all_same = all(np.array_equal(arrays[0], arr) for arr in arrays[1:])
    assert not all_same, "Images from different bands should have different data"
    
    print("✓ All multi-band retrieval tests passed!")
    return True


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("LSST Butler Mock Validation Tests")
    print("=" * 70)
    
    tests = [
        test_basic_butler_operations,
        test_geometry_operations,
        test_image_cutout_operations,
        test_wcs_transformations,
        test_multi_band_retrieval,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 All tests passed! The LSST Butler mocks are working correctly.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
