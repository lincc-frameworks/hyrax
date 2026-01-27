import os

import psutil

from hyrax import Hyrax

try:
    from hyrax.data_sets.data_cache import DataCache
except ImportError as e:
    raise NotImplementedError("No DataCache in this version") from e


class DataCacheBenchmarks:
    """Timing benchmarks for requesting data from the Hyrax random dataset"""

    def setup_cache(self):
        """Download CIFAR dataset only once"""
        self.h = Hyrax()

        # asv caches the cwd for each benchmark run
        self.h.config["general"]["results_dir"] = "./cifar"
        self.h.config["data_request"] = {
            "train": {
                "data": {
                    "dataset_class": "HyraxCifarDataset",
                    "data_location": "./cifar",
                    "fields": ["image", "label", "object_id"],
                }
            },
        }
        self.h.config["data_set"]["use_cache"] = True
        self.h.config["data_set"]["preload_cache"] = False
        self.data_provider = self.h.prepare()["train"]

    def setup(self):
        """
        Prepare for benchmark by defining and setting up the same random dataset
        Despite calling setup_cache this should not trigger another cifar download.
        """
        self.setup_cache()
        self.h.config["data_set"]["preload_cache"] = True

    def time_preload_cache_cifar(self):
        """Benchmark the amount of time needed to preload the cache of all data"""
        self.data_cache = DataCache(self.h.config, self.data_provider)
        self.data_cache.start_preload_thread()
        self.data_cache._preload_thread.join()

    def track_cache_cifar_hyrax_size_undercount(self):
        """Benchmark the amount of memory needed to preload the cache with Cifar data"""
        initial = psutil.Process(os.getpid()).memory_info().rss

        self.time_preload_cache_cifar()

        final = psutil.Process(os.getpid()).memory_info().rss
        os_size = final - initial
        hyrax_size = self.data_cache._data_size_bytes
        hyrax_undercount = os_size - hyrax_size
        return (hyrax_undercount / os_size) * 100

    track_cache_cifar_hyrax_size_undercount.units = "percent"
