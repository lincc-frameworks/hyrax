from pathlib import Path

import numpy as np

from hyrax import Hyrax


def test_save_to_database(loopback_inferred_hyrax):
    """Test that the data inserted into the vector database is not corrupted. i.e.
    that we can match ids to input vectors for all values."""

    h, dataset, inference_results = loopback_inferred_hyrax
    inference_result_ids = np.array(inference_results.ids())
    original_dataset_ids = np.array(dataset["infer"].ids())

    dataset = dataset["infer"]

    h.config["vector_db"]["name"] = "chromadb"
    original_shape = h.config["data_set"]["HyraxRandomDataset"]["shape"]

    # Populate the vector database with the results of inference
    vdb_path = h.config["general"]["results_dir"]
    h.save_to_database(output_dir=vdb_path)

    # Get a connection to the database that was just created.
    db_connection = h.database_connection(database_dir=vdb_path)

    # Verify that every inserted vector id matches the original vector
    for id in inference_result_ids:
        # Since the ordering of inference results is not guaranteed to match the
        # original dataset, we need to find the index of the original dataset id
        # that corresponds to the inference result id.
        assert id in original_dataset_ids, f"Inference ID, {id} not found in original dataset IDs."
        orig_indx = int(np.where(original_dataset_ids == id)[0][0])
        result = db_connection.get_by_id(id)
        saved_value = result[id].reshape(original_shape)
        original_value = dataset[orig_indx]["data"]["image"]
        assert np.all(np.isclose(saved_value, original_value))


def test_save_to_database_lance(tmp_path):
    """Test the full save_to_database → search workflow with the Lance backend.

    Lance requires ≥256 vectors to train its IVF/PQ index, so this test sets up
    its own Hyrax instance with a large-enough dataset rather than relying on the
    small ``loopback_inferred_hyrax`` fixture.

    For Lance the output_dir passed to save_to_database must be the same
    directory that holds lance_db/ (i.e. the timestamped infer results subdir).
    """
    h = Hyrax()
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True
    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "train_data"),
                "primary_id_field": "object_id",
                "split_fraction": 0.6,
            },
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "infer_data"),
                "primary_id_field": "object_id",
            },
        },
    }
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 300
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]  # 6-D vectors, matches loopback fixture

    weights_file = tmp_path / "fakeweights"
    weights_file.touch()
    h.config["infer"]["model_weights_file"] = str(weights_file)

    inference_results = h.infer()

    h.config["vector_db"]["name"] = "lance"
    h.config["vector_db"]["lance"]["num_partitions"] = 4  # safe with 300 vectors
    h.config["vector_db"]["lance"]["num_sub_vectors"] = 2  # divides 6-D vectors evenly

    # For Lance the output_dir is the timestamped infer-results subdir, which
    # already contains lance_db/.
    infer_results_dir = inference_results.data_location
    h.save_to_database(output_dir=infer_results_dir)

    db_connection = h.database_connection(database_dir=infer_results_dir)
    all_ids = np.array(inference_results.ids())

    for id in all_ids[:5]:  # spot-check 5 entries
        result = db_connection.get_by_id(id)
        assert id in result, f"ID {id} not returned by get_by_id"


def test_save_to_database_tensorboard_logging(loopback_inferred_hyrax):
    """Test that Tensorboard logs are created during vector database insertion."""

    h, dataset, inference_results = loopback_inferred_hyrax
    h.config["vector_db"]["name"] = "chromadb"

    # Populate the vector database with the results of inference
    vdb_path = h.config["general"]["results_dir"]
    h.save_to_database(output_dir=vdb_path)

    # Check that Tensorboard event files were created in the output directory
    tensorboard_files = list(Path(vdb_path).glob("events.out.tfevents.*"))
    assert len(tensorboard_files) > 0, "No Tensorboard event files found in output directory"

    # Optionally, we could parse the event files to check for our specific metrics
    # but that would require additional dependencies, so we'll just check for file existence
