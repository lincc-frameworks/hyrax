import numpy as np
import pytest

import hyrax
from hyrax.vector_dbs.chromadb_impl import ChromaDB


@pytest.fixture(scope="function", params=["RandomDataset"])  # , "RandomIterableDataset"])
def loopback_hyrax(tmp_path_factory, request):
    """This generates a loopback hyrax instance which is configured to use the
    loopback model and a simple dataset yielding random numbers. It includes a call
    to hyrax.infer which will produce the output consumed by vdb_index."""
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "LoopbackModel"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = 5
    h.config["general"]["results_dir"] = str(tmp_path_factory.mktemp(f"loopback_hyrax_{request.param}"))

    h.config["general"]["dev_mode"] = True
    h.config["data_set"]["name"] = request.param
    h.config["data_set"]["size"] = 20
    h.config["data_set"]["seed"] = 0

    h.config["data_set"]["validate_size"] = 0.2
    h.config["data_set"]["test_size"] = 0.2
    h.config["data_set"]["train_size"] = 0.6

    dataset = h.prepare()
    h.train()
    inference_results = h.infer()

    return h, dataset, inference_results


def test_vdb_index(loopback_hyrax):
    """Test that the data inserted into the vector database is not corrupted. i.e.
    that we can match ids to input vectors for all values."""

    h, dataset, inference_results = loopback_hyrax
    inference_result_ids = list(inference_results.ids())
    original_dataset_ids = list(dataset.ids())

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
