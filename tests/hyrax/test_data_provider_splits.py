"""Unit tests for DataProvider class with data splits.

This test module focuses on testing the DataProvider class in conjunction
with data splits (train/validate/test). It covers various logical paths
for defining and using data splits with DataProvider instances.
"""

from hyrax import Hyrax
from hyrax.data_sets.data_provider import DataProvider
from hyrax.pytorch_ignite import create_splits, dist_data_loader, setup_dataset


class TestDataProviderWithSplits:
    """Tests for DataProvider interaction with data split configurations."""

    def test_data_provider_with_train_validate_splits(self):
        """Test that DataProvider works correctly with train and validate splits."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./test_data",
                    "primary_id_field": "object_id",
                    "fields": ["image", "label"],
                    "dataset_config": {
                        "shape": [2, 8, 8],
                        "size": 100,
                        "seed": 42,
                    },
                }
            }
        }

        h.config["data_set"]["train_size"] = 0.6
        h.config["data_set"]["validate_size"] = 0.2
        h.config["data_set"]["test_size"] = 0.2
        h.config["data_set"]["seed"] = 42

        # Create DataProvider for train split
        dp = DataProvider(h.config, h.config["data_request"]["train"])

        # Verify DataProvider is properly initialized
        assert len(dp) == 100
        assert dp.primary_dataset == "data"

        # Create splits
        splits = create_splits(dp, h.config)

        # Verify split sizes
        assert len(splits["train"]) == 60
        assert len(splits["validate"]) == 20
        assert len(splits["test"]) == 20

        # Verify indices don't overlap
        train_set = set(splits["train"])
        validate_set = set(splits["validate"])
        test_set = set(splits["test"])

        assert len(train_set & validate_set) == 0
        assert len(train_set & test_set) == 0
        assert len(validate_set & test_set) == 0

    def test_data_provider_with_dist_data_loader_single_split(self):
        """Test DataProvider with dist_data_loader for a single split."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "random_data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./test_data",
                    "primary_id_field": "object_id",
                    "fields": ["image"],
                    "dataset_config": {
                        "shape": [2, 8, 8],
                        "size": 50,
                        "seed": 123,
                    },
                }
            }
        }

        h.config["data_set"]["train_size"] = 0.7
        h.config["data_set"]["test_size"] = 0.3
        h.config["data_set"]["validate_size"] = False  # Disable validate split
        h.config["data_set"]["seed"] = 123
        h.config["data_loader"]["batch_size"] = 4
        h.config["data_loader"]["num_workers"] = 0

        dp = DataProvider(h.config, h.config["data_request"]["train"])

        # Create data loader for train split
        loader, indices = dist_data_loader(dp, h.config, "train")

        # Verify we got the right number of samples
        assert len(indices) == 35  # 70% of 50

        # Verify loader is created
        assert loader is not None

    def test_data_provider_with_dist_data_loader_multiple_splits(self):
        """Test DataProvider with dist_data_loader for multiple splits."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "dataset_a": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./test_data_a",
                    "primary_id_field": "object_id",
                    "fields": ["image", "label"],
                    "dataset_config": {
                        "shape": [3, 16, 16],
                        "size": 120,
                        "seed": 456,
                    },
                }
            }
        }

        h.config["data_set"]["train_size"] = 0.6
        h.config["data_set"]["validate_size"] = 0.2
        h.config["data_set"]["test_size"] = 0.2
        h.config["data_set"]["seed"] = 456
        h.config["data_loader"]["batch_size"] = 8
        h.config["data_loader"]["num_workers"] = 0

        dp = DataProvider(h.config, h.config["data_request"]["train"])

        # Create data loaders for multiple splits
        loaders = dist_data_loader(dp, h.config, ["train", "validate"])

        # Verify we got loaders for both splits
        assert "train" in loaders
        assert "validate" in loaders

        train_loader, train_indices = loaders["train"]
        validate_loader, validate_indices = loaders["validate"]

        # Verify split sizes
        assert len(train_indices) == 72  # 60% of 120
        assert len(validate_indices) == 24  # 20% of 120

        # Verify loaders are created
        assert train_loader is not None
        assert validate_loader is not None

    def test_data_provider_multimodal_with_splits(self):
        """Test DataProvider with multiple datasets and splits."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "images": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./images",
                    "primary_id_field": "object_id",
                    "fields": ["image"],
                    "dataset_config": {
                        "shape": [3, 32, 32],
                        "size": 80,
                        "seed": 789,
                    },
                },
                "labels": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./labels",
                    "fields": ["label"],
                    "dataset_config": {
                        "shape": [1],
                        "size": 80,
                        "seed": 789,
                    },
                },
            }
        }

        h.config["data_set"]["train_size"] = 0.7
        h.config["data_set"]["validate_size"] = 0.15
        h.config["data_set"]["test_size"] = 0.15
        h.config["data_set"]["seed"] = 789
        h.config["data_loader"]["batch_size"] = 4
        h.config["data_loader"]["num_workers"] = 0

        # Create multimodal DataProvider
        dp = DataProvider(h.config, h.config["data_request"]["train"])

        # Verify both datasets are prepared
        assert len(dp.prepped_datasets) == 2
        assert "images" in dp.prepped_datasets
        assert "labels" in dp.prepped_datasets

        # Create splits
        splits = create_splits(dp, h.config)

        # Verify split sizes
        assert len(splits["train"]) == 56  # 70% of 80
        assert len(splits["validate"]) == 12  # 15% of 80
        assert len(splits["test"]) == 12  # 15% of 80

        # Test data retrieval from split indices
        for idx in splits["train"][:5]:  # Test first 5 samples
            data = dp[idx]
            assert "images" in data
            assert "labels" in data
            assert "object_id" in data

    def test_setup_dataset_creates_separate_data_providers_for_splits(self):
        """Test that setup_dataset creates separate DataProvider instances for each split."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "train_data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./train",
                    "primary_id_field": "object_id",
                    "fields": ["image"],
                    "dataset_config": {
                        "shape": [2, 16, 16],
                        "size": 100,
                        "seed": 111,
                    },
                }
            },
            "validate": {
                "validate_data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./validate",
                    "primary_id_field": "object_id",
                    "fields": ["image"],
                    "dataset_config": {
                        "shape": [2, 16, 16],
                        "size": 50,
                        "seed": 222,
                    },
                }
            },
            "infer": {
                "infer_data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./infer",
                    "primary_id_field": "object_id",
                    "fields": ["image"],
                    "dataset_config": {
                        "shape": [2, 16, 16],
                        "size": 30,
                        "seed": 333,
                    },
                }
            },
        }

        # Create dataset dictionary with separate DataProviders
        datasets = setup_dataset(h.config)

        # Verify separate DataProvider instances were created
        assert "train" in datasets
        assert "validate" in datasets
        assert "infer" in datasets

        # Verify each is a DataProvider instance
        assert isinstance(datasets["train"], DataProvider)
        assert isinstance(datasets["validate"], DataProvider)
        assert isinstance(datasets["infer"], DataProvider)

        # Verify correct sizes
        assert len(datasets["train"]) == 100
        assert len(datasets["validate"]) == 50
        assert len(datasets["infer"]) == 30

    def test_data_provider_collate_with_splits(self):
        """Test that DataProvider's collate function works correctly with split data."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./test_data",
                    "primary_id_field": "object_id",
                    "fields": ["image", "label"],
                    "dataset_config": {
                        "shape": [2, 8, 8],
                        "size": 60,
                        "seed": 444,
                    },
                }
            }
        }

        h.config["data_set"]["train_size"] = 0.5
        h.config["data_set"]["test_size"] = 0.5
        h.config["data_set"]["validate_size"] = False  # Disable validate split
        h.config["data_set"]["seed"] = 444

        dp = DataProvider(h.config, h.config["data_request"]["train"])
        splits = create_splits(dp, h.config)

        # Get a batch from train split
        batch_size = 5
        train_batch = [dp[idx] for idx in splits["train"][:batch_size]]

        # Collate the batch
        collated = dp.collate(train_batch)

        # Verify collated structure
        assert "data" in collated
        assert "image" in collated["data"]
        assert "label" in collated["data"]
        assert "object_id" in collated

        # Verify batch dimension
        import numpy as np

        assert isinstance(collated["data"]["image"], np.ndarray)
        assert len(collated["data"]["image"]) == batch_size

    def test_data_provider_with_integer_split_sizes(self):
        """Test DataProvider with integer split sizes instead of fractions."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./test_data",
                    "primary_id_field": "object_id",
                    "fields": ["image"],
                    "dataset_config": {
                        "shape": [2, 8, 8],
                        "size": 100,
                        "seed": 555,
                    },
                }
            }
        }

        # Use integer counts instead of fractions
        h.config["data_set"]["train_size"] = 60
        h.config["data_set"]["validate_size"] = 20
        h.config["data_set"]["test_size"] = 20
        h.config["data_set"]["seed"] = 555

        dp = DataProvider(h.config, h.config["data_request"]["train"])
        splits = create_splits(dp, h.config)

        # Verify split sizes match integer specifications
        assert len(splits["train"]) == 60
        assert len(splits["validate"]) == 20
        assert len(splits["test"]) == 20

    def test_data_provider_with_no_validate_split(self):
        """Test DataProvider when validate split is not configured."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./test_data",
                    "primary_id_field": "object_id",
                    "fields": ["image"],
                    "dataset_config": {
                        "shape": [2, 8, 8],
                        "size": 100,
                        "seed": 666,
                    },
                }
            }
        }

        h.config["data_set"]["train_size"] = 0.7
        h.config["data_set"]["test_size"] = 0.3
        h.config["data_set"]["validate_size"] = False  # Disable validate split
        h.config["data_set"]["seed"] = 666

        dp = DataProvider(h.config, h.config["data_request"]["train"])
        splits = create_splits(dp, h.config)

        # Verify only train and test splits exist
        assert "train" in splits
        assert "test" in splits
        assert "validate" not in splits

        # Verify sizes
        assert len(splits["train"]) == 70
        assert len(splits["test"]) == 30

    def test_data_provider_with_seeded_splits(self):
        """Test that DataProvider splits are reproducible with same seed."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./test_data",
                    "primary_id_field": "object_id",
                    "fields": ["image"],
                    "dataset_config": {
                        "shape": [2, 8, 8],
                        "size": 100,
                        "seed": 777,
                    },
                }
            }
        }

        h.config["data_set"]["train_size"] = 0.6
        h.config["data_set"]["validate_size"] = 0.2
        h.config["data_set"]["test_size"] = 0.2
        h.config["data_set"]["seed"] = 999  # Fixed seed for splits

        dp = DataProvider(h.config, h.config["data_request"]["train"])

        # Create splits twice with same seed
        splits1 = create_splits(dp, h.config)
        splits2 = create_splits(dp, h.config)

        # Verify splits are identical
        assert splits1["train"] == splits2["train"]
        assert splits1["validate"] == splits2["validate"]
        assert splits1["test"] == splits2["test"]

    def test_data_provider_metadata_with_splits(self):
        """Test that DataProvider metadata methods work correctly with split indices."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./test_data",
                    "primary_id_field": "object_id",
                    "fields": ["image"],
                    "dataset_config": {
                        "shape": [2, 8, 8],
                        "size": 50,
                        "seed": 888,
                    },
                }
            }
        }

        h.config["data_set"]["train_size"] = 0.6
        h.config["data_set"]["test_size"] = 0.4
        h.config["data_set"]["validate_size"] = False  # Disable validate split
        h.config["data_set"]["seed"] = 888

        dp = DataProvider(h.config, h.config["data_request"]["train"])
        splits = create_splits(dp, h.config)

        # Get metadata for train split indices
        train_metadata_fields = dp.metadata_fields("data")

        # Verify metadata fields are available
        assert "object_id" in train_metadata_fields

        # Get metadata for specific indices from train split
        train_indices = splits["train"][:5]
        metadata = dp.metadata(idxs=train_indices, fields=["meta_field_1_data"])

        # Verify metadata was retrieved correctly
        assert len(metadata) == 5

    def test_data_provider_ids_with_splits(self):
        """Test that DataProvider ids method works with split indices."""
        h = Hyrax()
        h.config["data_request"] = {
            "train": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "./test_data",
                    "primary_id_field": "object_id",
                    "fields": ["image"],
                    "dataset_config": {
                        "shape": [2, 8, 8],
                        "size": 40,
                        "seed": 1000,
                    },
                }
            }
        }

        h.config["data_set"]["train_size"] = 0.75
        h.config["data_set"]["test_size"] = 0.25
        h.config["data_set"]["validate_size"] = False  # Disable validate split
        h.config["data_set"]["seed"] = 1000

        dp = DataProvider(h.config, h.config["data_request"]["train"])
        splits = create_splits(dp, h.config)

        # Get all IDs
        all_ids = list(dp.ids())

        # Verify IDs can be accessed for split indices
        train_ids = [all_ids[idx] for idx in splits["train"][:5]]

        # Verify we got IDs
        assert len(train_ids) == 5
        assert all(isinstance(id_val, str) for id_val in train_ids)
