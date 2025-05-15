import numpy as np

from hyrax.vector_dbs.chromadb_impl import ChromaDB


def test_vdb_index(loopback_inferred_hyrax):
    """Test that the data inserted into the vector database is not corrupted. i.e.
    that we can match ids to input vectors for all values."""

    h, dataset, inference_results = loopback_inferred_hyrax
    inference_result_ids = list(inference_results.ids())
    original_dataset_ids = list(dataset.ids())

    h.config["vector_db"]["name"] = "chromadb"

    # Populate the vector database with the results of inference
    vdb_path = h.config["general"]["results_dir"]
    h.index(output_dir=vdb_path)

    chromadb_instance = ChromaDB({}, {"results_dir": vdb_path})
    chromadb_instance.connect()

    # Verify that every inserted vector id matches the original vector
    for indx, id in enumerate(inference_result_ids):
        assert id == original_dataset_ids[indx]
        result = chromadb_instance.get_by_id(id)
        original_value = dataset[indx]
        assert np.all(result[id] == original_value.numpy())
