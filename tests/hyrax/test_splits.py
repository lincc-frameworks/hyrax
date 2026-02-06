import pytest

from hyrax.pytorch_ignite import create_splits, dist_data_loader, load_split_indexes, save_split_indexes


def mkconfig(train_size=0.2, test_size=0.6, validate_size=0.1, seed=False):
    """Makes a configuration that has enough keys for create_splits"""
    return {
        "data_set": {
            "seed": seed,
            "train_size": train_size,
            "test_size": test_size,
            "validate_size": validate_size,
        },
    }


def test_split():
    """Test splitting in the default config where train, test, and validate are all specified"""

    fake_dataset = [1] * 100
    config = mkconfig()
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["validate"]) == 10
    assert len(indexes["test"]) == 60
    assert len(indexes["train"]) == 20


def test_split_no_validate():
    """Test splitting when validate is overridden"""
    fake_dataset = [1] * 100
    config = mkconfig(validate_size=False)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 60
    assert len(indexes["train"]) == 20
    assert indexes.get("validate") is None


def test_split_with_validate_no_test():
    """Test splitting when validate is provided by test size is not"""
    fake_dataset = [1] * 100
    config = mkconfig(test_size=False, validate_size=0.2)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["validate"]) == 20
    assert len(indexes["test"]) == 60
    assert len(indexes["train"]) == 20


def test_split_with_validate_no_test_no_train():
    """Test splitting when validate is provided by test size is not"""
    fake_dataset = [1] * 100
    config = mkconfig(test_size=False, train_size=False, validate_size=0.2)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 55
    assert len(indexes["train"]) == 25
    assert len(indexes["validate"]) == 20


def test_split_with_validate_with_test_no_train():
    """Test splitting when validate is provided by test size is not"""
    fake_dataset = [1] * 100
    config = mkconfig(test_size=0.6, train_size=False, validate_size=0.2)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 60
    assert len(indexes["train"]) == 20
    assert len(indexes["validate"]) == 20


def test_split_no_validate_no_test():
    """Test splitting when validate and test are overridden"""
    fake_dataset = [1] * 100
    config = mkconfig(validate_size=False, test_size=False)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 80
    assert len(indexes["train"]) == 20
    assert indexes.get("validate") is None


def test_split_no_validate_no_train():
    """Test splitting when validate and train are overridden"""
    fake_dataset = [1] * 100
    config = mkconfig(validate_size=False, train_size=False)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 60
    assert len(indexes["train"]) == 40
    assert indexes.get("validate") is None


def test_split_invalid_ratio():
    """Test that split RuntimeErrors when provided with an invalid ratio"""
    fake_dataset = [1] * 100

    with pytest.raises(RuntimeError):
        create_splits(fake_dataset, mkconfig(validate_size=False, train_size=1.1))

    with pytest.raises(RuntimeError):
        create_splits(fake_dataset, mkconfig(validate_size=False, train_size=-0.1))


def test_split_no_splits_configured():
    """Test splitting when all splits are overriden, and nothing is specified."""
    fake_dataset = [1] * 100
    config = mkconfig(validate_size=False, test_size=False, train_size=False)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 75
    assert len(indexes["train"]) == 25
    assert indexes.get("validate") is None


def test_split_values_configured():
    """Test splitting when all splits are integer data counts"""

    fake_dataset = [1] * 100
    config = mkconfig(validate_size=22, test_size=56, train_size=22)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 56
    assert len(indexes["train"]) == 22
    assert len(indexes["validate"]) == 22


def test_split_values_configured_no_validate():
    """Test splitting when all splits are integer data counts and validate is not configured
    so the total selected data doesn't cover the dataset.
    """
    fake_dataset = [1] * 100
    config = mkconfig(test_size=56, train_size=22)
    indexes = create_splits(fake_dataset, config)

    assert len(indexes["test"]) == 56
    assert len(indexes["train"]) == 22
    assert len(indexes["validate"]) == 10


def test_split_invalid_configured():
    """Test that split RuntimeErrors when provided with an invalid datapoint count"""
    fake_dataset = [1] * 100

    with pytest.raises(RuntimeError):
        create_splits(fake_dataset, mkconfig(validate_size=False, train_size=120))

    with pytest.raises(RuntimeError):
        create_splits(fake_dataset, mkconfig(validate_size=False, train_size=-10))


def test_split_values_rng():
    """Generate twice with the same RNG seed, verify same values are selected."""
    fake_dataset = [1] * 100
    config = mkconfig(test_size=56, train_size=22, seed=5)
    indexes_a = create_splits(fake_dataset, config)
    indexes_b = create_splits(fake_dataset, config)

    assert all([a == b for a, b in zip(indexes_a, indexes_b)])


def test_save_and_load_split_indexes(tmp_path):
    """Test saving and loading split indexes."""
    fake_dataset = [1] * 100
    config = mkconfig(test_size=0.6, train_size=0.2, validate_size=0.1, seed=42)

    # Create splits
    indexes = create_splits(fake_dataset, config)

    # Save splits
    save_split_indexes(indexes, tmp_path)

    # Verify individual files were created
    assert (tmp_path / "train.npy").exists()
    assert (tmp_path / "test.npy").exists()
    assert (tmp_path / "validate.npy").exists()

    # Load splits
    loaded_indexes = load_split_indexes(tmp_path)

    # Verify all splits are present
    assert set(loaded_indexes.keys()) == set(indexes.keys())

    # Verify all indexes match
    for split_name in indexes:
        assert loaded_indexes[split_name] == indexes[split_name]


def test_save_split_indexes_creates_directory(tmp_path):
    """Test that save_split_indexes creates the output directory if it doesn't exist."""
    fake_dataset = [1] * 100
    config = mkconfig(seed=42)

    # Create splits
    indexes = create_splits(fake_dataset, config)

    # Save to a non-existent subdirectory
    output_dir = tmp_path / "subdir" / "nested"
    save_split_indexes(indexes, output_dir)

    # Verify the files were created
    assert (output_dir / "train.npy").exists()
    assert (output_dir / "test.npy").exists()
    assert (output_dir / "validate.npy").exists()


def test_load_split_indexes_file_not_found(tmp_path):
    """Test that load_split_indexes raises FileNotFoundError when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_split_indexes(tmp_path)


def test_save_and_load_split_indexes_no_validate(tmp_path):
    """Test saving and loading split indexes when validate is not present."""
    fake_dataset = [1] * 100
    config = mkconfig(validate_size=False, seed=42)

    # Create splits
    indexes = create_splits(fake_dataset, config)

    # Save splits
    save_split_indexes(indexes, tmp_path)

    # Verify files were created
    assert (tmp_path / "train.npy").exists()
    assert (tmp_path / "test.npy").exists()
    assert not (tmp_path / "validate.npy").exists()

    # Load splits
    loaded_indexes = load_split_indexes(tmp_path, ["train", "test"])

    # Verify validate is not in loaded indexes
    assert "validate" not in loaded_indexes

    # Verify other splits match
    assert loaded_indexes["train"] == indexes["train"]
    assert loaded_indexes["test"] == indexes["test"]


def test_dist_data_loader_with_preloaded_indexes(tmp_path):
    """Test that dist_data_loader can use pre-loaded indexes."""
    import hyrax
    from hyrax.pytorch_ignite import create_splits, load_split_indexes, save_split_indexes, setup_dataset

    # Create a Hyrax instance with random dataset
    h = hyrax.Hyrax()
    h.config["data_loader"]["batch_size"] = 4
    h.config["data_loader"]["shuffle"] = False
    h.config["data_loader"]["num_workers"] = 0
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True

    h.config["model_inputs"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data_train"),
                "primary_id_field": "object_id",
            }
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data_infer"),
                "primary_id_field": "object_id",
            }
        },
    }

    h.config["data_set"]["HyraxRandomDataset"]["size"] = 50
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]
    h.config["data_set"]["train_size"] = 0.6
    h.config["data_set"]["validate_size"] = 0.2
    h.config["data_set"]["test_size"] = 0.2
    h.config["data_set"]["seed"] = 42

    # Create dataset using the same logic as in training
    dataset = setup_dataset(h.config)
    train_dataset = dataset["train"]

    # Generate splits normally
    original_indexes = create_splits(train_dataset, h.config)

    # Save the splits
    save_split_indexes(original_indexes, tmp_path)

    # Load the splits back
    loaded_indexes = load_split_indexes(tmp_path)

    # Use the loaded indexes with dist_data_loader
    data_loaders = dist_data_loader(train_dataset, h.config, ["train", "validate"], indexes=loaded_indexes)

    # Extract the indexes from the data loaders
    train_loader, train_indexes = data_loaders["train"]
    validate_loader, validate_indexes = data_loaders["validate"]

    # Verify that the indexes match the original
    assert train_indexes == original_indexes["train"]
    assert validate_indexes == original_indexes["validate"]

    # Verify the sizes are correct
    assert len(train_indexes) == 30  # 60% of 50
    assert len(validate_indexes) == 10  # 20% of 50
