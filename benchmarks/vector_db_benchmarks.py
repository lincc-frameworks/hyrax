import tempfile
from pathlib import Path

from hyrax import Hyrax


class VectorDBInsertBenchmarks:
    """Benchmarks for Hyrax vector database insert operations."""

    timeout = 120  # max seconds per benchmark before timing out

    # Parameters for the benchmarks: vector lengths and vector database implementations
    params = ([64, 256, 2048, 16_384], ["chromadb", "qdrant"])
    param_names = ["vector_length", "vector_db_implementation"]

    # Ideally this would be a `setup_cache` method, but `setup_cache` cannot be
    # parameterized by ASV. So instead we repeatedly call `setup` before each benchmark
    def setup(self, vector_length, vector_db_implementation):
        """Set up for vector database benchmarks. Create a temporary directory,
        configure Hyrax with a loopback model, and generate a random dataset, run
        inference to create the result files for insertion into the vector database."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.tmp_dir.name)

        self.h = Hyrax()
        self.h.config["general"]["results_dir"] = str(self.input_dir)
        self.h.config["model_inputs"] = {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "fields": ["image", "label", "object_id"],
                "primary_id_field": "object_id",
            }
        }
        self.h.config["model"]["name"] = "HyraxLoopback"

        # Default inference batch size is 512, so this should result in 4 batch files
        self.h.config["data_set"]["HyraxRandomDataset"]["size"] = 2048
        self.h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
        self.h.config["data_set"]["HyraxRandomDataset"]["shape"] = [vector_length]

        # Qdrant requires the vector size in order to create its collections
        self.h.config["vector_db"]["qdrant"]["vector_size"] = vector_length

        weights_file = self.input_dir / "fakeweights"
        with open(weights_file, "a"):
            pass
        self.h.config["infer"]["model_weights_file"] = str(weights_file)

        self.h.config["vector_db"]["name"] = vector_db_implementation

        self.h.infer()

    def tear_down(self):
        """Clean up the temporary directory used to store inference results."""
        self.tmp_dir.cleanup()

    def time_load_vector_db(self, vector_length, vector_db_implementation):
        """Timing benchmark for loading a vector database."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.h.save_to_database(output_dir=Path(tmp_dir))

    def peakmem_load_vector_db(self, vector_length, vector_db_implementation):
        """Memory benchmark for loading a vector database."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.h.save_to_database(output_dir=Path(tmp_dir))


class VectorDBSearchBenchmarks:
    """Benchmarks for Hyrax vector database search operations."""

    timeout = 120  # max seconds per benchmark before timing out

    # Parameters for the benchmarks: shard size limits and vector database implementations
    # The smaller shard size limit will result in parallelized searches, while the
    # larger shard size limit will trigger a sequential search across shards.
    params = ([64, 128], ["chromadb", "qdrant"])
    param_names = ["shard_size_limit", "vector_db_implementation"]

    def setup(self, shard_size_limit, vector_db_implementation):
        """Set up for vector database benchmarks. Create a temporary directory,
        configure Hyrax with a loopback model, and generate a random dataset, run
        inference to create the result files for insertion into the vector database."""
        self.tmp_input_dir = tempfile.TemporaryDirectory()
        self.tmp_output_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.tmp_input_dir.name)
        self.output_dir = Path(self.tmp_output_dir.name)

        self.vector_length = 1024

        self.h = Hyrax()
        self.h.config["general"]["results_dir"] = str(self.input_dir)
        self.h.config["model_inputs"] = {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "fields": ["image", "label", "object_id"],
                "primary_id_field": "object_id",
            }
        }
        self.h.config["data_loader"]["batch_size"] = 4096
        self.h.config["model"]["name"] = "HyraxLoopback"

        # Default inference batch size is 512, so this should result in 4 batch files
        self.h.config["data_set"]["HyraxRandomDataset"]["size"] = 4096
        self.h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
        self.h.config["data_set"]["HyraxRandomDataset"]["shape"] = [1024]

        # Create a fake weights file and then run inference on the random dataset
        weights_file = self.input_dir / "fakeweights"
        with open(weights_file, "a"):
            pass
        self.h.config["infer"]["model_weights_file"] = str(weights_file)

        self.h.infer()

        # Get the list of dataset ids
        self.ds = self.h.prepare()
        self.data_sample = self.ds[0]["data"]["image"]

        self.h.config["vector_db"]["name"] = vector_db_implementation
        self.h.config["vector_db"]["chromadb"]["shard_size_limit"] = shard_size_limit
        # Qdrant requires the vector size in order to create its collections
        self.h.config["vector_db"]["qdrant"]["vector_size"] = self.vector_length

        # Save inference results to vector database and create a db connection
        self.h.save_to_database(output_dir=Path(self.output_dir))
        self.db = self.h.database_connection(self.output_dir)

    def tear_down(self):
        """Clean up the temporary directory used to store inference results."""
        self.tmp_input_dir.cleanup()
        self.tmp_output_dir.cleanup()

    def time_search_by_vector_many_shards(self, shard_size_limit, vector_db_implementation):
        """Benchmark timing to perform a search by ID on a dataset with many shards."""
        self.db.search_by_vector(self.data_sample, k=1)

    def peakmem_search_by_vector_many_shards(self, shard_size_limit, vector_db_implementation):
        """Benchmark memory to perform a search by ID on a dataset with many shards."""
        self.db.search_by_vector(self.data_sample, k=1)
