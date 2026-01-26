"""
Unit tests for LSST dataset classes using Butler mocks.

This test module demonstrates how to use the mock LSST Butler objects
to test LSSTDataset and DownloadedLSSTDataset without requiring actual
LSST Science Pipelines or a Butler repository.
"""

import mocks
import pytest
import torch
import torchvision  # noqa: F401  # Import before mock contexts to prevent kernel re-registration
from mocks import lsst_config, mock_lsst_environment, sample_catalog, sample_catalog_saved  # noqa: F401

from hyrax.data_sets.downloaded_lsst_dataset import DownloadedLSSTDataset


class DownloadedLSSTDataSetInterruption(BaseException):
    """A class that will act like KeyboardInterrupt in a testing scenario,
    but isn't KeyboardInterrupt so it is distinguishable from a Ctrl-c during a test"""

    pass


class DownloadedLSSTDatasetMocked(DownloadedLSSTDataset):
    """This is a wrapper class around DownloadedLSSTDataset
    It performs major functions related to DownloadedLSSTDataset's multithreaded implementation, where
    _download_single_cutout() is the start of each worker thread.

    1. Keyboard interrupts are scriptable via the interrupt_after argument. After some count of
       _download_single_cutout() calls a ctrl-C will be simulated within a worker thread

    2. A custom patcher and keyword arguments to the patcher can be passed and they will be re-created on
       every call to _download_single_cutout(), thereby allowing the lsst mock environment to be
       re-created inside of every thread at startup.

    3. Resets the patch cache so tests always start with a clear cache.
    """

    def __init__(self, *args, interrupt_after=0, patcher=None, patcher_kwargs=None, **kwargs):
        self.patcher = patcher
        self.patcher_kwargs = {} if patcher_kwargs is None else patcher_kwargs
        self.interrupt_after = interrupt_after
        self.num_downloads = 0
        super().__init__(*args, **kwargs)
        self.clear_cache()

    def _download_single_cutout(self, *args):
        # print(f"Starting DL thread, {args}, {threading.current_thread().ident}")
        if self.interrupt_after != 0 and self.num_downloads >= self.interrupt_after:
            raise DownloadedLSSTDataSetInterruption("Download Interrupted")

        with self.patcher(**self.patcher_kwargs):
            self.num_downloads += 1
            return super()._download_single_cutout(*args)


def test_init(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """Test LSSTDataset initialization and basic operations with mocks.

    This test demonstrates how to use the mocks to test the LSSTDataset class
    without requiring actual LSST infrastructure.
    """
    with mock_lsst_environment():
        # Create LSSTDataset instance
        dataset = DownloadedLSSTDataset(lsst_config, data_location=str(tmp_path))

        # Verify catalog was loaded
        assert dataset.catalog is not None
        assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

        # Test basic dataset properties
        assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

        # Test fetching a cutout via the butler.
        data_record = dataset[0]
        cutout = data_record["data"]["image"]

        # Verify cutout is a tensor
        assert isinstance(cutout, torch.Tensor)

        # Verify it has the right number of bands (channels)
        assert cutout.shape[0] == 3  # g, r, i bands

        # Verify it is the first thing in the sample catalog
        assert dataset.metadata([0], ["object_id"])[0][0] == 1001


def test_download(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """
    Test LSSTDataset download golden path, verifying data access works when butler
    is disconnected.
    """
    with mock_lsst_environment():
        # Create LSSTDataset instance
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config, data_location=str(tmp_path), patcher=mock_lsst_environment
        )
        _manifest = dataset.download_cutouts()

    # Verify that download progress is correct:
    download_progress = dataset.download_progress()
    assert download_progress["progress_percent"] == 100.0
    assert download_progress["total"] == mocks.SAMPLE_CATALOG_LENGTH
    assert download_progress["successful"] == mocks.SAMPLE_CATALOG_LENGTH
    assert download_progress["failed"] == 0
    assert download_progress["pending"] == 0
    assert download_progress["completed"] == mocks.SAMPLE_CATALOG_LENGTH
    assert download_progress["remaining"] == 0

    # Verify catalog was loaded
    assert dataset.catalog is not None
    assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

    # Test basic dataset properties
    assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

    # Test fetching a cutout with the butler unavailable.
    data_record = dataset[0]
    cutout = data_record["data"]["image"]

    # Verify cutout is a tensor
    assert isinstance(cutout, torch.Tensor)

    # Verify it has the right number of bands (channels)
    assert cutout.shape[0] == 3  # g, r, i bands


def test_download_band_filtering(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """
    Test LSSTDataset band filtering. When a download is finished a catalog with fewer
    bands access should work properly.
    """
    with mock_lsst_environment():
        # Create LSSTDataset instance
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config, data_location=str(tmp_path), patcher=mock_lsst_environment
        )
        _manifest = dataset.download_cutouts()

    # Filter the bands
    lsst_config["data_set"]["filters"] = ["g", "r"]
    dataset = DownloadedLSSTDatasetMocked(
        lsst_config, data_location=str(tmp_path), patcher=mock_lsst_environment
    )

    # Verify catalog was loaded
    assert dataset.catalog is not None
    assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

    # Test basic dataset properties
    assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

    # Test fetching a cutout with the butler unavailable.
    data_record = dataset[0]
    cutout = data_record["data"]["image"]

    # Verify cutout is a tensor
    assert isinstance(cutout, torch.Tensor)

    # Verify it has the right number of bands (channels)
    assert cutout.shape[0] == 2  # g, r bands only


def test_interrupted_download(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """
    Test LSSTDataset can handle an interrupted download, and only gives access
    to the downloaded stuff.
    """
    with mock_lsst_environment():
        # Create LSSTDataset instance
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
            interrupt_after=3,
        )
        with pytest.raises(DownloadedLSSTDataSetInterruption):
            _manifest = dataset.download_cutouts()

    assert dataset.cache_info().misses == 3
    assert dataset.cache_info().hits == 0

    # Verify that download progress is correct:
    download_progress = dataset.download_progress()
    assert download_progress["progress_percent"] == 6.0
    assert download_progress["total"] == mocks.SAMPLE_CATALOG_LENGTH
    assert download_progress["successful"] == 3
    assert download_progress["failed"] == 0
    assert download_progress["pending"] == mocks.SAMPLE_CATALOG_LENGTH - 3
    assert download_progress["completed"] == 3
    assert download_progress["remaining"] == mocks.SAMPLE_CATALOG_LENGTH - 3

    # Verify catalog was loaded
    assert dataset.catalog is not None
    assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

    # Test basic dataset properties
    assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

    # Test fetching a cutout we downloaded with the butler unavailable.
    data_record = dataset[2]
    cutout = data_record["data"]["image"]

    # Verify cutout is a tensor
    assert isinstance(cutout, torch.Tensor)

    # Verify it has the right number of bands (channels)
    assert cutout.shape[0] == 3  # g, r, i bands

    # Verify access of an un-downloaded cutout raises the correct error.
    with pytest.raises(RuntimeError, match="un-downloaded cutout"):
        _ = dataset[4]


def test_interrupted_download_completes(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """
    Test LSSTDataset can handle an interrupted download, and that the download can finish after.
    """
    with mock_lsst_environment():
        # Create LSSTDataset instance
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
            interrupt_after=3,
        )
        with pytest.raises(DownloadedLSSTDataSetInterruption):
            _manifest = dataset.download_cutouts()

        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
        )
        _manifest = dataset.download_cutouts()

    # Verify catalog was loaded
    assert dataset.catalog is not None
    assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

    # Test basic dataset properties
    assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

    # Test fetching the last cutout we downloaded with the butler unavailable.
    data_record = dataset[mocks.SAMPLE_CATALOG_LENGTH - 1]
    cutout = data_record["data"]["image"]

    # Verify cutout is a tensor
    assert isinstance(cutout, torch.Tensor)

    # Verify it has the right number of bands (channels)
    assert cutout.shape[0] == 3  # g, r, i bands


def test_failed_download(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """
    Test LSSTDataset can handle a download where some tracts are failing, and only gives access
    to the downloaded stuff.
    """
    with mock_lsst_environment():
        # Create LSSTDataset instance
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
            patcher_kwargs={"fail_after_n": 7},
        )
        _manifest = dataset.download_cutouts()

    # Check that cache operated correctly
    assert dataset.cache_info().misses == mocks.NUM_TRACTS
    assert dataset.cache_info().hits == mocks.SAMPLE_CATALOG_LENGTH - mocks.NUM_TRACTS

    # Verify that download progress report is correct:
    download_progress = dataset.download_progress()
    assert download_progress["progress_percent"] == 18.0
    assert download_progress["total"] == mocks.SAMPLE_CATALOG_LENGTH
    assert download_progress["successful"] == 9
    assert download_progress["failed"] == 41
    assert download_progress["partial_downloads"] == 3
    assert download_progress["pending"] == 0
    assert download_progress["completed"] == mocks.SAMPLE_CATALOG_LENGTH
    assert download_progress["remaining"] == 0

    # Verify catalog was loaded
    assert dataset.catalog is not None
    assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

    # Test basic dataset properties
    assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

    # Test fetching a cutout we downloaded with the butler unavailable.
    data_record = dataset[2]
    cutout = data_record["data"]["image"]

    # Verify cutout is a tensor
    assert isinstance(cutout, torch.Tensor)

    # Verify it has the right number of bands (channels)
    assert cutout.shape[0] == 3  # g, r, i bands

    # Verify access of an un-downloaded cutout raises the correct error.
    with pytest.raises(RuntimeError, match="un-downloaded cutout"):
        _ = dataset[4]


def test_failed_download_completes(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """
    Test LSSTDataset can handle a failed download, and that the download can finish after.
    """
    with mock_lsst_environment():
        # Create first LSSTDataset instance to do a partial download
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
            patcher_kwargs={"fail_after_n": 34},
        )
        _ = dataset.download_cutouts()

        # Create second LSSTDataset instance to resume the download (no more failure this time)
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
            # patcher_kwargs={"fail_after_n": 7}
        )
        _manifest = dataset.download_cutouts(force_retry=True)

    # Verify catalog was loaded
    assert dataset.catalog is not None
    assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

    # Test basic dataset properties
    assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

    # Test fetching a cutout we downloaded with the butler unavailable.
    data_record = dataset[mocks.SAMPLE_CATALOG_LENGTH - 1]
    cutout = data_record["data"]["image"]

    # Verify cutout is a tensor
    assert isinstance(cutout, torch.Tensor)

    # Verify it has the right number of bands (channels)
    assert cutout.shape[0] == 3  # g, r, i bands


def test_failed_download_completes_on_reset(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """
    Test LSSTDataset can handle a failed download, and that the download can finish after
    using the reset_failed_downloads() function.
    """
    with mock_lsst_environment():
        # Create first LSSTDataset instance to do a partial download
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
            patcher_kwargs={"fail_after_n": 34},
        )
        _manifest = dataset.download_cutouts()

        # Create second LSSTDataset instance to resume the download (no more failure this time)
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
        )
        dataset.reset_failed_downloads()
        _manifest = dataset.download_cutouts()

    # Verify catalog was loaded
    assert dataset.catalog is not None
    assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

    # Test basic dataset properties
    assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

    # Test fetching a cutout we downloaded with the butler unavailable.
    data_record = dataset[mocks.SAMPLE_CATALOG_LENGTH - 1]
    cutout = data_record["data"]["image"]

    # Verify cutout is a tensor
    assert isinstance(cutout, torch.Tensor)

    # Verify it has the right number of bands (channels)
    assert cutout.shape[0] == 3  # g, r, i bands


def test_failed_band_download(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """
    Test LSSTDataset can handle a download where some bands are failing, and only gives access
    to the downloaded stuff.
    """
    with mock_lsst_environment():
        # Create LSSTDataset instance
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
            patcher_kwargs={"band_fail_after_n": {"g": 10}},
        )
        _manifest = dataset.download_cutouts()

    # Verify catalog was loaded
    assert dataset.catalog is not None
    assert len(dataset.catalog) == mocks.SAMPLE_CATALOG_LENGTH

    # Test basic dataset properties
    assert len(dataset) == mocks.SAMPLE_CATALOG_LENGTH

    # Test fetching a cutout where the G band is missing
    data_record = dataset[11]
    cutout = data_record["data"]["image"]

    # Verify cutout is a tensor
    assert isinstance(cutout, torch.Tensor)

    # Verify it has the right number of bands (channels)
    assert cutout.shape[0] == 3  # g, r, i bands

    # Verify it has all NaNs in the G channel
    assert torch.all(cutout[0] != cutout[0])

    # Verify it has all numbers in R,I channels
    assert torch.all(cutout[1] == cutout[1])
    assert torch.all(cutout[2] == cutout[2])


def test_band_detection_with_partial_downloads(mock_lsst_environment, lsst_config, tmp_path):  # noqa: F811
    """
    Test that _get_available_bands_from_manifest correctly identifies bands
    from complete downloads, ignoring partial downloads that may appear earlier
    in the manifest.
    """
    # Configure 4 bands
    lsst_config["data_set"]["filters"] = ["g", "r", "i", "z"]

    with mock_lsst_environment():
        # Make g and r bands fail for the FIRST 5 downloads each, then succeed
        # Early entries will have only i,z (partial), later entries will have all 4
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
            patcher_kwargs={"band_fail_before_n": {"g": 5, "r": 5}},
        )
        _manifest = dataset.download_cutouts()

    # Request only g,r,i bands - triggers _get_available_bands_from_manifest
    # which must find complete 4-band entries to determine available bands
    lsst_config["data_set"]["filters"] = ["g", "r", "i"]

    dataset = DownloadedLSSTDatasetMocked(
        lsst_config, data_location=str(tmp_path), patcher=mock_lsst_environment
    )

    # Verify band filtering found complete entries and set up correctly
    assert dataset._is_filtering_bands is True
    assert set(dataset.BANDS) == {"g", "r", "i"}


def test_catalog_ordering(mock_lsst_environment, lsst_config, tmp_path, sample_catalog):  # noqa: F811
    """
    Test that after a download the ordering of a new dataset object is given in the same order
    of the catalog that is provided, regardless of the catalog ordering or content of the downloading
    object.

    We do this by creating a download dir and then examining order between the original download object
    and a new object with the same data and manifest, but a truncated and permuted catalog.
    """
    import numpy as np

    with mock_lsst_environment():
        # Create LSSTDataset instance
        dataset = DownloadedLSSTDatasetMocked(
            lsst_config,
            data_location=str(tmp_path),
            patcher=mock_lsst_environment,
        )
        # Download the whole catalog
        _manifest = dataset.download_cutouts()

    # Create a copy of our standard sample catalog we will eventually use to filter
    # the downloaded dataset
    catalog_data = mocks.sample_catalog_data()

    # Get data for the first object
    first_object_id = catalog_data["object_id"][0]
    assert first_object_id == dataset.metadata([0], ["object_id"])[0][0]

    # Truncate the sample catalog
    catalog_truncation_index = int(mocks.SAMPLE_CATALOG_LENGTH / 2.0)
    for k, v in catalog_data.items():
        catalog_data[k] = v[0:catalog_truncation_index]

    # Permute the sample catalog ensuring the first object does not end up in the
    # 0th index
    rng = np.random.RandomState(mocks.SAMPLE_CATALOG_LENGTH)
    catalog_permutation = np.array(range(0, catalog_truncation_index))
    rng.shuffle(catalog_permutation)
    catalog_permutation = [int(i) for i in list(catalog_permutation)]
    for k, v in catalog_data.items():
        catalog_data[k] = list(np.array(v)[catalog_permutation])

    # Save the catalog out
    catalog_type, _ = sample_catalog
    catalog_table = mocks.convert_sample_catalog(catalog_type, catalog_data)
    catalog_path = mocks.save_catalog(catalog_type, catalog_table, tmp_path, "truncated_catalog")

    # Make a new dataset for access using the truncated and reordered catalog
    mocks.set_catalog(lsst_config, catalog_type, catalog_path)
    filtered_dataset = DownloadedLSSTDatasetMocked(
        lsst_config,
        data_location=str(tmp_path),
        patcher=mock_lsst_environment,
    )

    # New dataset should throw if we access outside our new catalog
    with pytest.raises(IndexError):
        _ = filtered_dataset[catalog_truncation_index + 1]

    # Data and ID of all objects should be in the correct index
    # given the permutation to the catalog that the filtered dataset is using
    for index, value in enumerate(catalog_permutation):
        # index indexes the filtered dataset
        # value is the index in the original dataset
        assert torch.all(filtered_dataset[index]["data"]["image"] == dataset[value]["data"]["image"])
        assert (
            filtered_dataset.metadata([index], ["object_id"])[0][0]
            == dataset.metadata([value], ["object_id"])[0][0]
        )
