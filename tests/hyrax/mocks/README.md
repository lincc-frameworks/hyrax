# LSST Butler Mocks

This directory contains mock implementations of LSST Butler objects for testing purposes.

## Overview

The LSST Science Pipelines provide a Butler API for accessing astronomical data. These mocks allow testing of `LSSTDataset` and `DownloadedLSSTDataset` without requiring:
- LSST Science Pipelines installation
- Access to an actual Butler repository
- Real astronomical data

## Files

- `lsst_butler_mocks.py` - Mock implementations of LSST Butler classes
- `__init__.py` - Package initialization

## Mock Classes

### Core Butler Classes

#### `MockButler`
Simulates `lsst.daf.butler.Butler` for data retrieval.

**Key Methods:**
- `get(dataset_type, data_id)` - Returns mock data products
  - `dataset_type="skyMap"` - Returns `MockSkyMap`
  - `dataset_type="deep_coadd"` - Returns `MockExposure` with image data

**Example:**
```python
butler = MockButler(repo="/fake/repo", collections="fake_collection")
skymap = butler.get("skyMap", {"skymap": "test_skymap"})
```

#### `MockSkyMap`
Simulates LSST SkyMap for sky tessellation.

**Key Methods:**
- `findTract(sphere_point)` - Returns `MockTractInfo`

#### `MockTractInfo`
Represents a tract in the sky tessellation.

**Key Methods:**
- `getId()` - Returns tract ID (default: 9813)
- `findPatch(sphere_point)` - Returns `MockPatchInfo`

#### `MockPatchInfo`
Represents a patch within a tract.

**Key Attributes:**
- `sequential_index` - Patch index (default: 42)

**Key Methods:**
- `getWcs()` - Returns `MockWcs` for coordinate transformations
- `getOuterBBox()` - Returns `MockBox2I` bounding box

### Geometry Classes

#### `MockGeom`
Container for geometry classes (simulates `lsst.geom` module).

**Classes:**
- `Box2I` - Integer bounding box
- `Box2D` - Floating-point bounding box
- `SpherePoint` - Celestial coordinates (RA/Dec)
- `degrees` - Angular unit

#### `MockBox2I`
Integer bounding box for pixel coordinates.

**Key Methods:**
- `getWidth()` - Returns box width in pixels
- `getHeight()` - Returns box height in pixels
- `isEmpty()` - Checks if box has zero area

#### `MockBox2D`
Floating-point bounding box.

**Key Methods:**
- `getMin()` - Returns minimum corner [x, y]
- `getMax()` - Returns maximum corner [x, y]

#### `MockSpherePoint`
Celestial coordinates (RA/Dec).

**Key Methods:**
- `getLongitude()` - Returns RA as angle object
- `getLatitude()` - Returns Dec as angle object
- `offset(bearing, distance)` - Returns offset point

### Image Classes

#### `MockExposure`
Simulates LSST Exposure containing image data.

**Key Methods:**
- `getImage()` - Returns `MockImage`

#### `MockImage`
Simulates LSST Image with pixel data.

**Key Methods:**
- `getArray()` - Returns numpy array of pixel values
- `__getitem__(box)` - Extracts cutout using `MockBox2I`

#### `MockWcs`
Simulates World Coordinate System transformations.

**Key Methods:**
- `skyToPixel(sky_points)` - Converts sky coordinates to pixel coordinates

## Usage Examples

### Basic Butler Operations

```python
from tests.hyrax.mocks.lsst_butler_mocks import MockButler, MockGeom, MockSpherePoint

# Create butler
butler = MockButler(repo="/fake/repo", collections="fake_collection")

# Get skymap
skymap = butler.get("skyMap", {"skymap": "hsc_pdr3"})

# Find tract and patch for a sky position
degrees = MockGeom.degrees(1.0)
position = MockSpherePoint(150.0, 2.0, degrees)
tract_info = skymap.findTract(position)
patch_info = tract_info.findPatch(position)

# Retrieve image data
exposure = butler.get("deep_coadd", {
    "tract": tract_info.getId(),
    "patch": patch_info.sequential_index,
    "band": "g",
    "skymap": "hsc_pdr3",
})

# Extract image array
image = exposure.getImage()
array = image.getArray()  # numpy array of shape (10000, 10000)
```

### Testing with Mocks

```python
import unittest.mock as mock
from tests.hyrax.mocks.lsst_butler_mocks import MockButler, MockGeom

# Patch LSST modules
mock_butler_module = mock.MagicMock()
mock_butler_module.Butler = MockButler

mock_geom_module = mock.MagicMock()
mock_geom_module.Box2I = MockGeom.Box2I
mock_geom_module.Box2D = MockGeom.Box2D
mock_geom_module.SpherePoint = MockGeom.SpherePoint
mock_geom_module.degrees = MockGeom.degrees(1.0)

with mock.patch.dict('sys.modules', {
    'lsst.daf.butler': mock_butler_module,
    'lsst.geom': mock_geom_module,
}):
    # Now import and test LSSTDataset
    from hyrax.data_sets.lsst_dataset import LSSTDataset
    # ... test code ...
```

## Implementation Details

### Data Generation

- **Images**: Created with reproducible random data using numpy
  - Seed based on tract/patch/band for consistency
  - Default shape: (10000, 10000) pixels
  - Data type: float32

- **Tract/Patch IDs**: Fixed values for simplicity
  - Tract ID: 9813
  - Patch index: 42

### Coordinate Transformations

- **WCS**: Simple linear transformation
  - Pixel scale: 0.2 arcsec/pixel (1/18000 deg/pixel)
  - Reference point: RA=0, Dec=0 at pixel (50000, 50000)

- **SpherePoint.offset()**: Simplified offsets along cardinal directions
  - 0° (East), 90° (North), 180° (West), 270° (South)

### Cutout Operations

- **Image slicing**: `image[box]` extracts cutout using `MockBox2I`
- **Box containment**: All reasonable cutouts fit within 10000x10000 patch

## Testing

See `test_lsst_butler_mocks.py` for comprehensive unit tests demonstrating:
- Basic butler operations
- Geometry operations
- Image cutout extraction
- Multi-band image retrieval
- Integration with LSSTDataset and DownloadedLSSTDataset

Run validation tests:
```bash
cd tests/hyrax
python validate_mocks.py  # Standalone validation
pytest test_lsst_butler_mocks.py  # Full test suite
```

## Limitations

These mocks are designed for **unit testing only** and have the following limitations:

1. **Simplified geometry**: WCS transformations are linear approximations
2. **Fixed IDs**: Always returns same tract/patch IDs
3. **No edge cases**: Doesn't handle boundary conditions like real Butler
4. **Limited validation**: Doesn't check for invalid inputs
5. **Mock data only**: Returns random data, not real astronomical images

## Extension

To add support for additional Butler functionality:

1. Add new dataset types to `MockButler.get()`
2. Implement additional LSST classes as needed
3. Update tests to cover new functionality

## References

- [LSST Science Pipelines Documentation](https://pipelines.lsst.io/)
- [Butler Documentation](https://pipelines.lsst.io/middleware/butler.html)
- Original code: `src/hyrax/data_sets/lsst_dataset.py`
- Original code: `src/hyrax/data_sets/downloaded_lsst_dataset.py`
