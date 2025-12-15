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

import threading
import unittest.mock as mock

import numpy as np
import pytest

# Force the pretend images to all be small,
# we wrap around on some operations to ensure things like the wcs
# always give valid pixels for the image.
MOCK_IMAGE_MAX_SIZE = 1000
MOCK_CUTOUT_SIZE = 100


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
        if len(args) == 2 and hasattr(args[0], "getMin") and hasattr(args[0], "getMax"):
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
            self._max_x = MOCK_CUTOUT_SIZE
            self._max_y = MOCK_CUTOUT_SIZE

    def getWidth(self):  # noqa: N802
        """Return box width in pixels."""
        return self._max_x - self._min_x

    def getHeight(self):  # noqa: N802
        """Return box height in pixels."""
        return self._max_y - self._min_y

    def isEmpty(self):  # noqa: N802
        """Check if box has zero area."""
        return self.getWidth() <= 0 or self.getHeight() <= 0

    def contains(self, obj):
        """Mock Contains method. Returns that every box is inside every other box."""
        # Note, the only place this is used in LSSTDataset, the one box not containing the other is the error
        # So we return true. All boxes contain all other boxes!
        return True

    def __hash__(self):
        return hash((self._min_x, self._max_x, self._min_y, self._max_y))

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

    def getMin(self):  # noqa: N802
        """Return minimum corner point."""
        return [self._min_x, self._min_y]

    def getMax(self):  # noqa: N802
        """Return maximum corner point."""
        return [self._max_x, self._max_y]


class MockAngle:
    """Mock of the lsst angle class"""

    def __init__(self, value, units):  # noqa: D102
        if units is None or type(units).__name__ != "degrees":
            msg = "MockAngle: Must provide units and they must be mocked degrees."
            msg += f"You passed '{type(units).__name__}'"
            raise NotImplementedError(msg)

        self.value = value
        self.units = units

    def asDegrees(self):  # noqa: N802, D102
        return self.value


class MockSpherePoint:
    """Mock implementation of lsst.geom.SpherePoint for celestial coordinates."""

    def __init__(self, ra, dec, units):
        """Initialize sphere point with RA/Dec.

        Args:
            ra: Right ascension value
            dec: Declination value
            units: Angular units (mock degrees object)
        """
        if units is None or type(units).__name__ != "degrees":
            msg = "MockSpherePoint: Must provide units and they must be mocked degrees."
            msg += f"You passed '{type(units).__name__}'"
            raise NotImplementedError(msg)

        self._ra = MockAngle(ra, units)
        self._dec = MockAngle(dec, units)

    def getLongitude(self):  # noqa: N802
        """Return mock longitude/RA object."""
        return self._ra

    def getLatitude(self):  # noqa: N802
        """Return mock latitude/Dec object."""
        return self._dec

    def offset(self, bearing, distance):
        """Mock offset operation - returns a new point with small offset.

        Args:
            bearing: Direction in degrees (mock degrees object)
            distance: Angular distance (mock degrees object)

        Returns:
            New MockSpherePoint offset from this one
        """
        # Simple approximation - just add small offsets
        offset_deg = 0.01 if hasattr(distance, "asDegrees") else distance

        # Bearing determines direction
        bearing_val = bearing.asDegrees() if hasattr(bearing, "asDegrees") else bearing

        # Very simple offset based on bearing
        if bearing_val == 0.0:  # East in RA
            new_ra = self._ra.asDegrees() + offset_deg
            new_dec = self._dec.asDegrees()
        elif bearing_val == 90.0:  # North in Dec
            new_ra = self._ra.asDegrees()
            new_dec = self._dec.asDegrees() + offset_deg
        elif bearing_val == 180.0:  # West in RA
            new_ra = self._ra.asDegrees() - offset_deg
            new_dec = self._dec.asDegrees()
        elif bearing_val == 270.0:  # South in Dec
            new_ra = self._ra.asDegrees()
            new_dec = self._dec.asDegrees() - offset_deg
        else:
            new_ra = self._ra.asDegrees() + offset_deg * 0.5
            new_dec = self._dec.asDegrees() + offset_deg * 0.5

        return MockSpherePoint(new_ra, new_dec, _mock_degrees)


class MockWcs:
    """Mock implementation of WCS (World Coordinate System) for coordinate transformations."""

    def __init__(self):
        """Initialize with default transformation (simple linear mapping)."""
        # Simple scale: 0.2 arcsec/pixel = 1/18000 degrees/pixel
        self._pixel_scale = 1.0 / 18000.0  # degrees per pixel

    def skyToPixel(self, sky_points):  # noqa: N802
        """Convert sky coordinates to pixel coordinates.

        Args:
            sky_points: List of MockSpherePoint objects

        Returns:
            List of [x, y] pixel coordinate pairs
        """
        pixel_points = []
        for pt in sky_points:
            # Simple linear transformation centered at RA=0, Dec=0
            ra = pt.getLongitude().asDegrees()
            dec = pt.getLatitude().asDegrees()

            # Convert to pixels (arbitrary reference point)
            x = (ra / self._pixel_scale) % MOCK_IMAGE_MAX_SIZE
            y = (dec / self._pixel_scale) % MOCK_IMAGE_MAX_SIZE

            # Convert to pixels (arbitrary reference point)
            # x = ra / self._pixel_scale + 50000  # Offset to get positive pixels
            # y = dec / self._pixel_scale + 50000

            pixel_points.append([x, y])

        return pixel_points


class MockImage:
    """Mock implementation of LSST image with getArray() method."""

    def __init__(self, data):
        """Initialize mock image.

        Args:
            data: numpy array of image data, or None to create random data
            shape: Shape of image if data is None
        """
        self._data = data

    def getArray(self):  # noqa: N802
        """Return the underlying numpy array."""
        return self._data

    def __getitem__(self, box):
        """Support slicing with Box2I to extract cutout.

        Args:
            box: MockBox2I defining the region to extract

        Returns:
            New MockImage with sliced data. The data is randomly generated and
            does not match the tract at all, but is the right size.
        """
        if isinstance(box, MockBox2I):
            # Extract the region defined by the box
            seed = hash(box) % (2**32)
            rng = np.random.RandomState(seed)
            cutout_data = rng.randn(box.getHeight(), box.getWidth()).astype(np.float32)
            return MockImage(data=cutout_data)
        else:
            # Fallback to standard indexing
            return MockImage(data=self._data[box])

    def shape(self):
        """Pass through to the underlying image for quality of life debugging tests"""
        return self.data.shape


class MockExposure:
    """Mock implementation of LSST Exposure with image and WCS."""

    def __init__(self, image_data=None):
        """Initialize mock exposure.

        Args:
            image_data: numpy array for the image, or None
            shape: Shape if image_data is None
        """
        self._image = MockImage(data=image_data)

    def getImage(self):  # noqa: N802
        """Return the mock image."""
        return self._image

    def shape(self):
        """Pass through to the underlying image for quality of life debugging tests"""
        return self._image.shape


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
        self._outer_bbox._max_x = MOCK_IMAGE_MAX_SIZE
        self._outer_bbox._max_y = MOCK_IMAGE_MAX_SIZE

    def getWcs(self):  # noqa: N802
        """Return WCS for this patch."""
        return self._wcs

    def getOuterBBox(self):  # noqa: N802
        """Return outer bounding box of the patch."""
        return self._outer_bbox


class MockTractInfo:
    """Mock implementation of TractInfo for tract operations."""

    def __init__(self, tract_id=0, patch_id=0):
        """Initialize mock tract.

        Args:
            tract_id: Globaly Unique ID for this tract
            patch_id: Globaly Unique ID for the only patch in this tract
        """
        self._tract_id = tract_id
        self._patch_id = patch_id

    def getId(self):  # noqa: N802
        """Return tract ID."""
        return self._tract_id

    def findPatch(self, sphere_point):  # noqa: N802
        """Find patch containing the given sky position.

        Args:
            sphere_point: MockSpherePoint with coordinates

        Returns:
            MockPatchInfo for the patch
        """
        # Simple mock: always return same patch for now
        return MockPatchInfo(sequential_index=42)


NUM_TRACTS = 20


class MockSkyMap:
    """Mock implementation of LSST SkyMap for tract/patch lookups."""

    ids = [{"tract_id": 9813 + i, "patch_id": 42 + i} for i in range(NUM_TRACTS)]
    id_index = 0

    def __init__(self, name="mock_skymap"):
        """Initialize mock skymap.

        Args:
            name: Name identifier for the skymap
        """
        self._name = name

    def reset(self):
        """Reset the round-robin of patch lookups to ensure test isolation."""
        MockSkyMap.id_index = 0

    def findTract(self, sphere_point):  # noqa: N802
        """Find tract containing the given sky position.

        Args:
            sphere_point: MockSpherePoint with coordinates

        Returns:
            MockTractInfo for the tract
        """
        # Return a tract with IDs from our list,
        # each tract only has one patch
        retval = MockTractInfo(**MockSkyMap.ids[MockSkyMap.id_index])
        MockSkyMap.id_index += 1
        MockSkyMap.id_index %= len(MockSkyMap.ids)
        # print(retval.getId())
        return retval


class MockButler:
    """Mock implementation of LSST Butler for data retrieval.

    The Butler is the primary interface for accessing data in LSST pipelines.
    This mock simulates getting skymaps and image exposures.
    """

    initialized_thread_ids = []
    initialized_thread_ids_lock = threading.Lock()
    fail_prob = 0.0
    band_fail_prob = {}
    fail_after_n = 0
    band_fail_after_n = {}

    @classmethod
    def reset(cls, fail_prob=0.0, band_fail_prob=None, fail_after_n=0, band_fail_after_n=None):
        """Resets the mock butler for a new test, and configures failure behavior

        Parameters
        ----------
        fail_prob : float, optional
            How often should a butler get fail randomly, by default 0.0
        band_fail_prob : dict, optional
            How often should a butler get that does not fail becuase of fail_prob fail for a particular
            band access. Given by providing a dictionary of band -> probability. For example
            band_fail_prob={"g": 0.1} would cause gets to g band to fail 10% of the time, by default {}
        fail_after_n : int, optional
            Continually fail after the provided number of calls to butler get. The default of 0 disables get
            failures for this reason, but leaves probalistic failures configured above intact.
        band_fail_after_n : dict, optional
            Contiually fail particular band(s) after the provided number of calls to butler.get in the
            particular band. Dictionary provided has bands as keys and counts as values.
            Counts of zero mean no failures for that band
        """
        cls.initialized_thread_ids = []
        cls.fail_prob = fail_prob
        cls.band_fail_prob = {} if band_fail_prob is None else band_fail_prob
        cls.fail_after_n = fail_after_n
        cls.band_fail_after_n = {} if band_fail_after_n is None else band_fail_after_n

    def __init__(self, repo=None, collections=None):
        """Initialize mock butler.

        Args:
            repo: Repository path (unused in mock)
            collections: Collections to query (unused in mock)
        """
        self._repo = repo
        self._collections = collections
        self.request_count = 0
        self.band_request_count = {}

        # Ensure only one Mock Butler per thread
        thread_id = threading.current_thread().ident
        if thread_id in MockButler.initialized_thread_ids:
            msg = f"Cannot make two butlers on one thread tid:{thread_id}, "
            msg += f"initialized_tids {MockButler.initialized_thread_ids}."
            raise RuntimeError(msg)
        else:
            with MockButler.initialized_thread_ids_lock:
                MockButler.initialized_thread_ids.append(thread_id)

        # Store mock data that can be retrieved
        self._data = {}

    def _generate_errors(self, rng, band):
        if MockButler.fail_after_n != 0 and self.request_count >= MockButler.fail_after_n:
            msg = f"MockButler: Simulated fail after {self.request_count} requests."
            raise RuntimeError(msg)

        if rng.random() > 1.0 - MockButler.fail_prob:
            msg = f"MockButler: Simulated fail due to overall fail_prob = {MockButler.fail_prob}"
            raise RuntimeError(msg)

        band_limit = MockButler.band_fail_after_n.get(band, 0)
        if band_limit != 0 and self.band_request_count.get(band, 0) >= band_limit:
            msg = f"MockButler: Simulated fail after {band_limit} requests to {band} band."
            raise RuntimeError(msg)

        band_fail_prob = MockButler.band_fail_prob.get(band, 0.0)
        if rng.random() > 1.0 - band_fail_prob:
            msg = f"MockButler: Simulated fail due to band failure probability {band} = {band_fail_prob}"
            raise RuntimeError(msg)

        self.request_count += 1
        if self.band_request_count.get(band) is None:
            self.band_request_count[band] = 1
        else:
            self.band_request_count[band] += 1

    def get(self, dataset_type, data_id=None):
        """Retrieve mock data product.

        Args:
            dataset_type: Type of data to retrieve (e.g., "skyMap", "deep_coadd")
            data_id: Dictionary identifying which data to get

        Returns:
            Mock object depending on dataset_type
        """
        data_id = {} if data_id is None else data_id

        if dataset_type == "skyMap":
            # Return a mock skymap
            skymap_name = data_id.get("skymap", "mock") if data_id else "mock"
            return MockSkyMap(name=skymap_name)

        elif dataset_type == "deep_coadd":
            # Return a mock exposure with an image
            # Create unique but deterministic data based on tract/patch/band
            tract = data_id.get("tract", 0)
            patch = data_id.get("patch", 0)
            band = data_id.get("band", "g")

            # Create reproducible random data
            seed = hash((tract, patch, band)) % (2**32)
            rng = np.random.RandomState(seed)

            # Raise a simulated error probabilistically if configured at class level.
            self._generate_errors(rng, band)

            # Create image data - larger than typical cutouts
            image_data = rng.randn(MOCK_IMAGE_MAX_SIZE, MOCK_IMAGE_MAX_SIZE).astype(np.float32)

            return MockExposure(image_data=image_data)

        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")


# Mock geometry module that contains the classes
class MockGeom:
    """Mock of lsst.geom module containing geometry classes."""

    Box2I = MockBox2I
    Box2D = MockBox2D
    SpherePoint = MockSpherePoint

    class degrees:  # noqa N801
        """Mock degrees unit for angular measurements."""

        def __init__(self, value=1.0):
            self.value = value

        def asDegrees(self):  # noqa: N802, D102
            return self.value

        def asArcseconds(self):  # noqa: N802, D102
            return self.value * 3600.0

        def __mul__(self, other):
            return MockGeom.degrees(self.value * other)

        def __rmul__(self, other):
            return MockGeom.degrees(other * self.value)


# Convenience: make degrees a singleton-like
_mock_degrees = MockGeom.degrees(1.0)


@pytest.fixture
def mock_lsst_environment():
    """Fixture providing a complete mock LSST environment.

    This fixture returns a patcher for the lsst.daf.butler and lsst.geom modules
    to use our mock implementations, allowing tests to control when those mocks are
    in place. You use it in a test like so:

    .. code-block:: python

        def test_stuff(mock_lsst_environment):
            with mock_lsst_environment():
                # Do things with butler available
                pass

            # Do things without butler available


    The arguments optional arguments to the function are keyword args to ```MockButler.reset()``` that
    control the failure behavior of the mock butler. For example

    .. code-block:: python
        def test_stuff(mock_lsst_environment):
            with mock_lsst_environment(fail_prob = 0.1):
                # Do things with butler available, but 10% of butler gets fail.
                pass

            # Do things without butler available

    """

    def mock_lsst_context(**kwargs):
        """Keyword arguments"""
        # Create mock modules
        mock_geom_module = mock.MagicMock()
        mock_geom_module.Box2I = MockGeom.Box2I
        mock_geom_module.Box2D = MockGeom.Box2D
        mock_geom_module.SpherePoint = MockGeom.SpherePoint
        mock_geom_module.degrees = MockGeom.degrees(1.0)

        mock_butler_module = mock.MagicMock()
        mock_butler_module.Butler = MockButler
        MockButler.reset(**kwargs)

        mock_daf_module = mock.MagicMock()
        mock_daf_module.butler = mock_butler_module

        mock_lsst_module = mock.MagicMock()
        mock_lsst_module.daf = mock_daf_module
        mock_lsst_module.geom = mock_geom_module

        return mock.patch.dict(
            "sys.modules",
            {
                "lsst": mock_lsst_module,
                "lsst.daf": mock_daf_module,
                "lsst.daf.butler": mock_butler_module,
                "lsst.geom": mock_geom_module,
            },
        )

    return mock_lsst_context
