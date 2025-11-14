import unittest.mock as mock
from pathlib import Path
from typing import Any

import numpy as np
import torch
from astropy.table import Table

from hyrax.data_sets.downloaded_lsst_dataset import DownloadedLSSTDataset


class FakeDownloadedLSSTEnvironment:
    """
    Mock environment for DownloadedLSSTDataset testing.

    Creates fake manifest files and cutout files to test subset catalog functionality
    without requiring real LSST data or file I/O.
    """

    def __init__(self, full_object_ids: list, subset_object_ids: list):
        self.full_object_ids = full_object_ids
        self.subset_object_ids = subset_object_ids
        self.patchers: list[mock._patch[Any]] = []

        # Create fake manifest data
        self.fake_manifest = self._create_fake_manifest()

        # Create fake cutout data - each cutout contains its object_id for verification
        self.fake_cutouts = {
            obj_id: torch.tensor([[[float(obj_id)]]], dtype=torch.float32) for obj_id in full_object_ids
        }

    def _create_fake_manifest(self):
        """Create a fake manifest with all full_object_ids"""
        manifest_data = {
            "objectId": self.full_object_ids,
            "cutout_shape": [np.array([1, 1, 1], dtype=int) for _ in self.full_object_ids],
            "filename": [f"cutout_{obj_id}.pt" for obj_id in self.full_object_ids],
            "downloaded_bands": ["g" for _ in self.full_object_ids],  # Single band for simplicity
        }
        return Table(manifest_data)

    def _create_subset_catalog(self):
        """Create subset catalog with only subset_object_ids"""
        catalog_data = {
            "objectId": self.subset_object_ids,
            "ra": [0.0] * len(self.subset_object_ids),  # Dummy coordinates
            "dec": [0.0] * len(self.subset_object_ids),
        }
        return Table(catalog_data)

    def _mock_torch_load(self, path, **kwargs):
        """Mock torch.load to return fake cutout data based on filename"""
        filename = Path(path).name
        # Extract object_id from filename like "cutout_1003.pt"
        obj_id = int(filename.replace("cutout_", "").replace(".pt", ""))
        return self.fake_cutouts[obj_id]

    def _mock_torch_save(self, tensor, path):
        """Mock torch.save - just store what was saved for verification"""
        pass

    def _mock_manifest_read(self, path):
        """Mock reading manifest file"""
        return self.fake_manifest

    def _mock_manifest_write(self, path, **kwargs):
        """Mock writing manifest file"""
        pass

    def __enter__(self):
        # Mock torch operations
        self.patchers.append(mock.patch("torch.load", side_effect=self._mock_torch_load))
        self.patchers.append(mock.patch("torch.save", side_effect=self._mock_torch_save))

        # Mock manifest file operations
        self.patchers.append(mock.patch("astropy.table.Table.read", side_effect=self._mock_manifest_read))
        self.patchers.append(mock.patch("astropy.table.Table.write", side_effect=self._mock_manifest_write))

        # Mock Path.exists - create a simple function
        def mock_exists(path_instance):
            """Mock function for Path.exists() - path_instance is the Path object"""
            # Manifest file exists
            if path_instance.name in ["manifest.fits", "manifest.parquet"]:
                return True

            # Cutout files exist for all full_object_ids
            if path_instance.name.startswith("cutout_") and path_instance.name.endswith(".pt"):
                obj_id = int(path_instance.name.replace("cutout_", "").replace(".pt", ""))
                return obj_id in self.full_object_ids

            return False

        self.patchers.append(mock.patch("pathlib.Path.exists", mock_exists))

        # Mock Path.mkdir to avoid directory creation
        self.patchers.append(mock.patch("pathlib.Path.mkdir"))

        # Start all patches
        for patcher in self.patchers:
            patcher.start()

        return self

    def __exit__(self, *exc):
        # Stop all patches
        for patcher in self.patchers:
            patcher.stop()

    def get_subset_catalog(self):
        """Get the subset catalog for testing"""
        return self._create_subset_catalog()


def test_subset_catalog_preserves_manifest_and_loads_correct_cutouts():
    """
    Test that when a subset catalog is passed to DownloadedLSSTDataset:
    1. The full manifest is preserved (no overwriting)
    2. Only the subset objects are accessible via dataset indexing
    3. The correct cutouts are loaded for each subset object
    """
    # Test data: 10 objects in manifest, 3 in subset (non-consecutive to test index mapping)
    full_object_ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    subset_object_ids = [1003, 1007, 1009]

    with FakeDownloadedLSSTEnvironment(full_object_ids, subset_object_ids) as fake_env:
        # Create config for DownloadedLSSTDataset
        config = {
            "general": {"data_dir": "/fake/data/dir"},
            "data_set": {
                "butler_repo": "/fake/butler",
                "butler_collection": "fake_collection",
                "skymap": "fake_skymap",
                "object_id_column_name": "objectId",
                "use_cache": False,
                "preload_cache": False,
            },
        }

        # Create subset catalog
        subset_catalog = fake_env.get_subset_catalog()

        # Mock the parent class initialization and problematic methods during init
        with (
            mock.patch("hyrax.data_sets.downloaded_lsst_dataset.LSSTDataset.__init__"),
            mock.patch.object(DownloadedLSSTDataset, "_setup_naming_strategy"),
            mock.patch.object(DownloadedLSSTDataset, "_initialize_manifest"),
        ):
            # Create DownloadedLSSTDataset with mocked initialization
            dataset = DownloadedLSSTDataset(config, data_location=config["general"]["data_dir"])

            # Set required attributes after creation
            dataset.catalog = subset_catalog  # Set catalog BEFORE other methods need it
            dataset.BANDS = ("g",)  # Single band for simplicity
            dataset.download_dir = Path("/fake/data/dir")  # Required for manifest path
            dataset._butler_config = None  # Will trigger existing manifest path
            dataset._config = config  # Required for _setup_naming_strategy

            # Manually set naming strategy attributes instead of calling the method
            dataset.use_object_id = True
            dataset.object_id_column = "objectId"

            # Set band filtering attributes
            dataset._is_filtering_bands = False
            dataset._band_indices = None
            dataset._original_bands = ("g",)

            # Since the full initialization is complex, let's manually set up the test scenario
            # This focuses on testing the subset functionality rather than initialization
            dataset.manifest = fake_env.fake_manifest
            dataset.catalog_type = "astropy"
            dataset.manifest_path = dataset.download_dir / "manifest.fits"

            # Simulate the subset scenario - set up filtering manually
            dataset._manifest_filter_object_ids = set(subset_object_ids)
            dataset._build_catalog_to_manifest_index_map(dataset.manifest)
            print(f"Manifest length: {len(dataset.manifest)}")
            print(f"Catalog length: {len(dataset.catalog)}")
            print(f"Index mapping: {dataset._catalog_to_manifest_index_map}")

        # Verify the dataset length matches subset size
        assert len(dataset) == 3, f"Expected dataset length 3, got {len(dataset)}"

        # Verify the manifest still contains all original objects
        assert len(dataset.manifest) == 10, f"Expected manifest length 10, got {len(dataset.manifest)}"

        # Verify that filtering is active (subset scenario)
        assert dataset._catalog_to_manifest_index_map is not None, "Index mapping should be active for subset"
        assert dataset._manifest_filter_object_ids is not None, "Filtering should be active for subset"

        # Verify correct index mapping
        expected_mapping = {0: 2, 1: 6, 2: 8}  # catalog_idx -> manifest_idx
        assert dataset._catalog_to_manifest_index_map == expected_mapping, (
            f"Index mapping incorrect: {dataset._catalog_to_manifest_index_map}"
        )

        # Test that accessing dataset elements returns correct cutouts
        # Mock the apply_transform method to return tensors as-is for testing
        with mock.patch.object(dataset, "apply_transform", side_effect=lambda x: x):
            # dataset[0] should return cutout for object 1003
            cutout_0 = dataset[0]["data"]["image"]
            assert cutout_0.item() == 1003.0, f"dataset[0] should contain object 1003, got {cutout_0.item()}"

            # dataset[1] should return cutout for object 1007
            cutout_1 = dataset[1]["data"]["image"]
            assert cutout_1.item() == 1007.0, f"dataset[1] should contain object 1007, got {cutout_1.item()}"

            # dataset[2] should return cutout for object 1009
            cutout_2 = dataset[2]["data"]["image"]
            assert cutout_2.item() == 1009.0, f"dataset[2] should contain object 1009, got {cutout_2.item()}"

        # Verify that the manifest contains the expected object IDs in order
        manifest_object_ids = list(dataset.manifest["objectId"])
        assert manifest_object_ids == full_object_ids, (
            f"Manifest object IDs should be preserved: expected {full_object_ids}, got {manifest_object_ids}"
        )


def test_subset_catalog_with_band_filtering():
    """
    Test that band filtering works correctly with subset catalogs:
    1. When bands are filtered, only requested bands are returned
    2. Band filtering is applied to cached cutouts correctly
    3. Subset catalog functionality still works with band filtering
    """
    # Test data: 5 objects in manifest, 3 in subset, multiple bands
    full_object_ids = [2001, 2002, 2003, 2004, 2005]
    subset_object_ids = [2002, 2004, 2005]

    with FakeDownloadedLSSTEnvironment(full_object_ids, subset_object_ids) as fake_env:
        # Override the manifest to have multiple bands
        multi_band_manifest_data = {
            "objectId": full_object_ids,
            "cutout_shape": [np.array([3, 1, 1], dtype=int) for _ in full_object_ids],  # 3 bands
            "filename": [f"cutout_{obj_id}.pt" for obj_id in full_object_ids],
            "downloaded_bands": [["g", "r", "i"] for _ in full_object_ids],  # Multiple bands
        }
        fake_env.fake_manifest = Table(multi_band_manifest_data)

        # Create multi-band fake cutouts - each cutout has 3 bands with distinct values
        fake_env.fake_cutouts = {
            obj_id: torch.tensor(
                [[[float(obj_id)]], [[float(obj_id + 0.1)]], [[float(obj_id + 0.2)]]], dtype=torch.float32
            )
            for obj_id in full_object_ids
        }

        # Create config for DownloadedLSSTDataset with band filtering
        config = {
            "general": {"data_dir": "/fake/data/dir"},
            "data_set": {
                "butler_repo": "/fake/butler",
                "butler_collection": "fake_collection",
                "skymap": "fake_skymap",
                "object_id_column_name": "objectId",
                "bands": ["g", "i"],  # Filter to only g and i bands (indices 0 and 2)
                "use_cache": False,
                "preload_cache": False,
            },
        }

        # Create subset catalog
        subset_catalog = fake_env.get_subset_catalog()

        # Mock the parent class initialization and problematic methods during init
        with (
            mock.patch("hyrax.data_sets.downloaded_lsst_dataset.LSSTDataset.__init__"),
            mock.patch.object(DownloadedLSSTDataset, "_setup_naming_strategy"),
            mock.patch.object(DownloadedLSSTDataset, "_initialize_manifest"),
        ):
            # Create DownloadedLSSTDataset with mocked initialization
            dataset = DownloadedLSSTDataset(config, data_location=config["general"]["data_dir"])

            # Set required attributes after creation
            dataset.catalog = subset_catalog
            dataset.BANDS = ("g", "r", "i")  # All available bands
            dataset.download_dir = Path("/fake/data/dir")
            dataset._butler_config = None
            dataset._config = config

            # Manually set naming strategy attributes
            dataset.use_object_id = True
            dataset.object_id_column = "objectId"

            # Set band filtering attributes - filtering to g and i bands (indices 0, 2)
            dataset._is_filtering_bands = True
            dataset._band_indices = [0, 2]  # g and i bands
            dataset._original_bands = ("g", "r", "i")

            # Set up the test scenario
            dataset.manifest = fake_env.fake_manifest
            dataset.catalog_type = "astropy"
            dataset.manifest_path = dataset.download_dir / "manifest.fits"

            # Simulate the subset scenario
            dataset._manifest_filter_object_ids = set(subset_object_ids)
            dataset._build_catalog_to_manifest_index_map(dataset.manifest)
            print(f"Manifest length: {len(dataset.manifest)}")
            print(f"Catalog length: {len(dataset.catalog)}")
            print(f"Band filtering enabled: {dataset._is_filtering_bands}")
            print(f"Band indices: {dataset._band_indices}")
            print(f"Index mapping: {dataset._catalog_to_manifest_index_map}")

        # Verify basic setup
        assert len(dataset) == 3, f"Expected dataset length 3, got {len(dataset)}"
        assert len(dataset.manifest) == 5, f"Expected manifest length 5, got {len(dataset.manifest)}"

        # Verify index mapping is correct (catalog index -> manifest index)
        # Objects 2002, 2004, 2005 are at manifest indices 1, 3, 4
        expected_mapping = {0: 1, 1: 3, 2: 4}  # catalog_idx -> manifest_idx
        assert dataset._catalog_to_manifest_index_map == expected_mapping, (
            f"Index mapping incorrect: {dataset._catalog_to_manifest_index_map}"
        )

        # Test that accessing dataset elements returns filtered cutouts
        with mock.patch.object(dataset, "apply_transform", side_effect=lambda x: x):
            # dataset[0] should return cutout for object 2002 with only g and i bands
            cutout_0 = dataset[0]["data"]["image"]
            print(f"Cutout 0 shape: {cutout_0.shape}")
            print(f"Cutout 0 values: {cutout_0.flatten()}")

            # Should have 2 bands (g and i), not 3
            assert cutout_0.shape[0] == 2, f"Expected 2 bands after filtering, got {cutout_0.shape[0]}"

            # Should contain values for g band (2002.0) and i band (2002.2)
            expected_values = torch.tensor([2002.0, 2002.2])
            actual_values = cutout_0.flatten()
            assert torch.allclose(actual_values, expected_values), (
                f"Expected filtered values {expected_values}, got {actual_values}"
            )

            # dataset[1] should return cutout for object 2004 with band filtering
            cutout_1 = dataset[1]["data"]["image"]
            assert cutout_1.shape[0] == 2, f"Expected 2 bands after filtering, got {cutout_1.shape[0]}"

            expected_values_1 = torch.tensor([2004.0, 2004.2])  # g and i bands
            actual_values_1 = cutout_1.flatten()
            assert torch.allclose(actual_values_1, expected_values_1), (
                f"Expected filtered values {expected_values_1}, got {actual_values_1}"
            )

            # dataset[2] should return cutout for object 2005 with band filtering
            cutout_2 = dataset[2]["data"]["image"]
            assert cutout_2.shape[0] == 2, f"Expected 2 bands after filtering, got {cutout_2.shape[0]}"

            expected_values_2 = torch.tensor([2005.0, 2005.2])  # g and i bands
            actual_values_2 = cutout_2.flatten()
            assert torch.allclose(actual_values_2, expected_values_2), (
                f"Expected filtered values {expected_values_2}, got {actual_values_2}"
            )


if __name__ == "__main__":
    test_subset_catalog_preserves_manifest_and_loads_correct_cutouts()
    print("Test passed: Subset catalog correctly preserves manifest and loads correct cutouts")

    test_subset_catalog_with_band_filtering()
    print("Test passed: Band filtering works correctly with subset catalogs")
