import tempfile
from pathlib import Path

import numpy as np

from hyrax import Hyrax


class DatasetRequestBenchmarks:
    """Timing benchmarks for requesting data from the Hyrax random dataset"""

    def setup(self):
        """Prepare for benchmark by defining and setting up a random dataset"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.tmp_dir.name)

        self.h = Hyrax()
        self.h.config["general"]["results_dir"] = str(self.input_dir)
        self.h.config["model_inputs"] = {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(self.input_dir),
                "fields": ["image", "label", "object_id"],
            }
        }

        num_vectors = 4096
        self.h.config["data_set"]["HyraxRandomDataset"]["size"] = num_vectors
        self.h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
        self.h.config["data_set"]["HyraxRandomDataset"]["shape"] = [3, 64, 64]

        self.ds = self.h.prepare()

        self.indexes = np.random.randint(0, num_vectors, size=128, dtype=int)

    def time_request_all_data(self):
        """Benchmark the amount of time needed to retrieve all the data from
        the random dataset
        """
        for indx in self.indexes:
            self.ds[indx]
