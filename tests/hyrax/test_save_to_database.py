import numpy as np


def test_save_to_database(loopback_inferred_hyrax):
    """Test that the data inserted into the vector database is not corrupted. i.e.
    that we can match ids to input vectors for all values."""

    h, dataset, inference_results = loopback_inferred_hyrax
    inference_result_ids = list(inference_results.ids())
    original_dataset_ids = list(dataset.ids())

    h.config["vector_db"]["name"] = "chromadb"
    dim_1_length = h.config["data_set"]["dimension_1_length"]
    dim_2_length = h.config["data_set"]["dimension_2_length"]
    # Populate the vector database with the results of inference
    vdb_path = h.config["general"]["results_dir"]
    h.save_to_database(output_dir=vdb_path)

    # Get a connection to the database that was just created.
    db_connection = h.database_connection(database_dir=vdb_path)

    # Verify that every inserted vector id matches the original vector
    for indx, id in enumerate(inference_result_ids):
        assert id == original_dataset_ids[indx]
        result = db_connection.get_by_id(id)
        saved_value = result[id].reshape(dim_1_length, dim_2_length)
        original_value = dataset[indx]
        assert np.all(saved_value == original_value.numpy())
