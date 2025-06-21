import tempfile
from pathlib import Path

from hyrax import Hyrax


class VectorDBBenchmarks:
    """Benchmarks for Hyrax vector database operations."""

    timeout = 120  # max seconds per benchmark before timing out

    # Parameters for the benchmarks: vector lengths and vector database implementations
    params = ([64, 256, 2048, 16_384], ["chromadb"])
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
        self.h.config["data_set"]["name"] = "HyraxRandomDataset"
        self.h.config["model"]["name"] = "HyraxLoopback"

        # Default inference batch size is 512, so this should result in 4 batch files
        self.h.config["data_set.random_dataset"]["size"] = 2048
        self.h.config["data_set.random_dataset"]["seed"] = 0
        self.h.config["data_set.random_dataset"]["shape"] = [vector_length]

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
