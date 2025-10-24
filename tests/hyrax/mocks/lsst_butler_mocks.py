"""
Mock objects for LSST Butler testing.

This module provides mock implementations of LSST Butler objects to enable
unit testing of LSSTDataset and DownloadedLSSTDataset without requiring
the actual LSST Science Pipelines or access to a Butler repository.

The mocks simulate:
- Butler API for retrieving data
- SkyMap and tract/patch finding operations
- Image objects with WCS transformations
- Box geometry operations

Usage:
    from tests.hyrax.mocks.lsst_butler_mocks import MockButler, MockSkyMap
    
    butler = MockButler(repo="/fake/repo", collections="fake_collection")
    skymap = butler.get("skyMap", {"skymap": "fake_skymap"})
    tract_info = skymap.findTract(sphere_point)
"""

import numpy as np


class MockBox2I:
    """Mock implementation of lsst.geom.Box2I for bounding box operations."""
    
    EXPAND = "EXPAND"  # Expansion strategy constant
    
    def __init__(self, *args, **kwargs):
        """Initialize mock box with default or specified dimensions.
        
        Args can be:
        - Box2D, expansion_strategy
        - min_point, max_point
        - min_point, extent
        """
        if len(args) == 2 and hasattr(args[0], 'getMin') and hasattr(args[0], 'getMax'):
            # Created from Box2D
            box2d = args[0]
            min_pt = box2d.getMin()
            max_pt = box2d.getMax()
            self._min_x = int(min_pt[0])
            self._min_y = int(min_pt[1])
            self._max_x = int(max_pt[0])
            self._max_y = int(max_pt[1])
        else:
            # Default small box for testing
            self._min_x = 0
            self._min_y = 0
            self._max_x = 100
            self._max_y = 100
    
    def getWidth(self):
        """Return box width in pixels."""
        return self._max_x - self._min_x
    
    def getHeight(self):
        """Return box height in pixels."""
        return self._max_y - self._min_y
    
    def isEmpty(self):
        """Check if box has zero area."""
        return self.getWidth() <= 0 or self.getHeight() <= 0
    
    def __repr__(self):
        return f"MockBox2I(min=({self._min_x}, {self._min_y}), max=({self._max_x}, {self._max_y}))"


class MockBox2D:
    """Mock implementation of lsst.geom.Box2D for floating-point bounding boxes."""
    
    def __init__(self, *points):
        """Initialize from a list of points.
        
        Args:
            points: List of [x, y] coordinate pairs
        """
        if len(points) >= 2:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            self._min_x = min(xs)
            self._min_y = min(ys)
            self._max_x = max(xs)
            self._max_y = max(ys)
        else:
            self._min_x = 0.0
            self._min_y = 0.0
            self._max_x = 100.0
            self._max_y = 100.0
    
    def getMin(self):
        """Return minimum corner point."""
        return [self._min_x, self._min_y]
    
    def getMax(self):
        """Return maximum corner point."""
        return [self._max_x, self._max_y]


class MockSpherePoint:
    """Mock implementation of lsst.geom.SpherePoint for celestial coordinates."""
    
    def __init__(self, ra, dec, units):
        """Initialize sphere point with RA/Dec.
        
        Args:
            ra: Right ascension value
            dec: Declination value
            units: Angular units (mock degrees object)
        """
        self._ra = ra
        self._dec = dec
        self._units = units
    
    def getLongitude(self):
        """Return mock longitude/RA object."""
        class Angle:
            def __init__(self, value):
                self.value = value
            def asDegrees(self):
                return self.value
        return Angle(self._ra)
    
    def getLatitude(self):
        """Return mock latitude/Dec object."""
        class Angle:
            def __init__(self, value):
                self.value = value
            def asDegrees(self):
                return self.value
        return Angle(self._dec)
    
    def offset(self, bearing, distance):
        """Mock offset operation - returns a new point with small offset.
        
        Args:
            bearing: Direction in degrees (mock degrees object)
            distance: Angular distance (mock degrees object)
            
        Returns:
            New MockSpherePoint offset from this one
        """
        # Simple approximation - just add small offsets
        offset_deg = 0.01 if hasattr(distance, 'asDegrees') else distance
        
        # Bearing determines direction
        if hasattr(bearing, 'asDegrees'):
            bearing_val = bearing.asDegrees()
        else:
            bearing_val = bearing
            
        # Very simple offset based on bearing
        if bearing_val == 0.0:  # East in RA
            new_ra = self._ra + offset_deg
            new_dec = self._dec
        elif bearing_val == 90.0:  # North in Dec
            new_ra = self._ra
            new_dec = self._dec + offset_deg
        elif bearing_val == 180.0:  # West in RA
            new_ra = self._ra - offset_deg
            new_dec = self._dec
        elif bearing_val == 270.0:  # South in Dec
            new_ra = self._ra
            new_dec = self._dec - offset_deg
        else:
            new_ra = self._ra + offset_deg * 0.5
            new_dec = self._dec + offset_deg * 0.5
            
        return MockSpherePoint(new_ra, new_dec, self._units)


class MockWcs:
    """Mock implementation of WCS (World Coordinate System) for coordinate transformations."""
    
    def __init__(self):
        """Initialize with default transformation (simple linear mapping)."""
        # Simple scale: 0.2 arcsec/pixel = 1/18000 degrees/pixel
        self._pixel_scale = 1.0 / 18000.0  # degrees per pixel
    
    def skyToPixel(self, sky_points):
        """Convert sky coordinates to pixel coordinates.
        
        Args:
            sky_points: List of MockSpherePoint objects
            
        Returns:
            List of [x, y] pixel coordinate pairs
        """
        pixel_points = []
        for pt in sky_points:
            # Simple linear transformation centered at RA=0, Dec=0
            ra = pt._ra
            dec = pt._dec
            
            # Convert to pixels (arbitrary reference point)
            x = ra / self._pixel_scale + 50000  # Offset to get positive pixels
            y = dec / self._pixel_scale + 50000
            
            pixel_points.append([x, y])
        
        return pixel_points


class MockImage:
    """Mock implementation of LSST image with getArray() method."""
    
    def __init__(self, data=None, shape=(100, 100)):
        """Initialize mock image.
        
        Args:
            data: numpy array of image data, or None to create random data
            shape: Shape of image if data is None
        """
        if data is not None:
            self._data = data
        else:
            # Create random float data
            self._data = np.random.randn(*shape).astype(np.float32)
    
    def getArray(self):
        """Return the underlying numpy array."""
        return self._data
    
    def __getitem__(self, box):
        """Support slicing with Box2I to extract cutout.
        
        Args:
            box: MockBox2I defining the region to extract
            
        Returns:
            New MockImage with sliced data
        """
        if isinstance(box, MockBox2I):
            # Extract the region defined by the box
            y_slice = slice(box._min_y, box._max_y)
            x_slice = slice(box._min_x, box._max_x)
            cutout_data = self._data[y_slice, x_slice].copy()
            return MockImage(data=cutout_data)
        else:
            # Fallback to standard indexing
            return MockImage(data=self._data[box])


class MockExposure:
    """Mock implementation of LSST Exposure with image and WCS."""
    
    def __init__(self, image_data=None, shape=(100, 100)):
        """Initialize mock exposure.
        
        Args:
            image_data: numpy array for the image, or None
            shape: Shape if image_data is None
        """
        self._image = MockImage(data=image_data, shape=shape)
    
    def getImage(self):
        """Return the mock image."""
        return self._image


class MockPatchInfo:
    """Mock implementation of PatchInfo for tract/patch operations."""
    
    def __init__(self, sequential_index=0):
        """Initialize mock patch.
        
        Args:
            sequential_index: Unique index for this patch
        """
        self.sequential_index = sequential_index
        self._wcs = MockWcs()
        # Create outer bbox that contains all reasonable cutouts
        self._outer_bbox = MockBox2I()
        self._outer_bbox._min_x = 0
        self._outer_bbox._min_y = 0
        self._outer_bbox._max_x = 10000
        self._outer_bbox._max_y = 10000
    
    def getWcs(self):
        """Return WCS for this patch."""
        return self._wcs
    
    def getOuterBBox(self):
        """Return outer bounding box of the patch."""
        return self._outer_bbox


class MockTractInfo:
    """Mock implementation of TractInfo for tract operations."""
    
    def __init__(self, tract_id=0):
        """Initialize mock tract.
        
        Args:
            tract_id: Unique ID for this tract
        """
        self._tract_id = tract_id
    
    def getId(self):
        """Return tract ID."""
        return self._tract_id
    
    def findPatch(self, sphere_point):
        """Find patch containing the given sky position.
        
        Args:
            sphere_point: MockSpherePoint with coordinates
            
        Returns:
            MockPatchInfo for the patch
        """
        # Simple mock: always return same patch for now
        return MockPatchInfo(sequential_index=42)


class MockSkyMap:
    """Mock implementation of LSST SkyMap for tract/patch lookups."""
    
    def __init__(self, name="mock_skymap"):
        """Initialize mock skymap.
        
        Args:
            name: Name identifier for the skymap
        """
        self._name = name
    
    def findTract(self, sphere_point):
        """Find tract containing the given sky position.
        
        Args:
            sphere_point: MockSpherePoint with coordinates
            
        Returns:
            MockTractInfo for the tract
        """
        # Simple mock: always return same tract for now
        return MockTractInfo(tract_id=9813)


class MockButler:
    """Mock implementation of LSST Butler for data retrieval.
    
    The Butler is the primary interface for accessing data in LSST pipelines.
    This mock simulates getting skymaps and image exposures.
    """
    
    def __init__(self, repo=None, collections=None):
        """Initialize mock butler.
        
        Args:
            repo: Repository path (unused in mock)
            collections: Collections to query (unused in mock)
        """
        self._repo = repo
        self._collections = collections
        # Store mock data that can be retrieved
        self._data = {}
    
    def get(self, dataset_type, data_id=None):
        """Retrieve mock data product.
        
        Args:
            dataset_type: Type of data to retrieve (e.g., "skyMap", "deep_coadd")
            data_id: Dictionary identifying which data to get
            
        Returns:
            Mock object depending on dataset_type
        """
        if dataset_type == "skyMap":
            # Return a mock skymap
            skymap_name = data_id.get("skymap", "mock") if data_id else "mock"
            return MockSkyMap(name=skymap_name)
        
        elif dataset_type == "deep_coadd":
            # Return a mock exposure with an image
            # Create unique but deterministic data based on tract/patch/band
            tract = data_id.get("tract", 0) if data_id else 0
            patch = data_id.get("patch", 0) if data_id else 0
            band = data_id.get("band", "g") if data_id else "g"
            
            # Create reproducible random data
            seed = hash((tract, patch, band)) % (2**32)
            rng = np.random.RandomState(seed)
            
            # Create image data - larger than typical cutouts
            image_data = rng.randn(10000, 10000).astype(np.float32)
            
            return MockExposure(image_data=image_data)
        
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")


# Mock geometry module that contains the classes
class MockGeom:
    """Mock of lsst.geom module containing geometry classes."""
    
    Box2I = MockBox2I
    Box2D = MockBox2D
    SpherePoint = MockSpherePoint
    
    class degrees:
        """Mock degrees unit for angular measurements."""
        
        def __init__(self, value=1.0):
            self.value = value
        
        def asDegrees(self):
            return self.value
        
        def asArcseconds(self):
            return self.value * 3600.0
        
        def __mul__(self, other):
            return MockGeom.degrees(self.value * other)
        
        def __rmul__(self, other):
            return MockGeom.degrees(other * self.value)


# Convenience: make degrees a singleton-like
_mock_degrees = MockGeom.degrees(1.0)


def create_mock_butler_environment():
    """Create and return a complete mock LSST Butler environment.
    
    Returns:
        dict with 'butler' and 'geom' keys containing mock objects
    """
    return {
        'butler': MockButler(),
        'geom': MockGeom,
        'degrees': _mock_degrees,
    }
