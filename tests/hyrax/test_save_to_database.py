import numpy as np


def test_save_to_database(loopback_inferred_hyrax):
    """Test that the data inserted into the vector database is not corrupted. i.e.
    that we can match ids to input vectors for all values."""

    h, dataset, inference_results = loopback_inferred_hyrax
    inference_result_ids = np.array(list(inference_results.ids()))
    original_dataset_ids = np.array(list(dataset.ids()))

    # If the dataset is iterable, convert it to a list for easier indexing
    if dataset.is_iterable():
        dataset = list(dataset)

    h.config["vector_db"]["name"] = "chromadb"
    original_shape = h.config["data_set.random_dataset"]["shape"]

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
        orig_indx = np.where(original_dataset_ids == id)[0][0]
        result = db_connection.get_by_id(id)
        saved_value = result[id].reshape(original_shape)
        original_value = dataset[orig_indx]["image"]
        assert np.all(np.isclose(saved_value, original_value.numpy()))
