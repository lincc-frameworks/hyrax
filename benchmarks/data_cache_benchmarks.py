import os
from pathlib import Path

import psutil

from hyrax import Hyrax


class DataCacheBenchmarks:
    """Timing benchmarks for preloading the data cache with the HSC1k dataset."""

    def setup_cache(self):
        """Download the HSC1k dataset only once."""
        import pooch

        self.h = Hyrax()

        # asv caches the cwd for each benchmark run
        data_dir = Path("./hsc1k").resolve()
        data_dir.mkdir(exist_ok=True)
        pooch.retrieve(
            # Zenodo URL for example HSC dataset
            url="https://zenodo.org/records/14498536/files/hsc_demo_data.zip?download=1",
            known_hash="md5:1be05a6b49505054de441a7262a09671",
            fname="hsc_demo_data.zip",
            path=data_dir,
            processor=pooch.Unzip(extract_dir=str(data_dir)),
        )
        # Extracted folder name from the bundled HSC1k sample dataset.
        hsc_data_dir = data_dir / "hsc_8asec_1000"

        self.h.config["general"]["results_dir"] = str(data_dir)
        self.h.config["general"]["data_dir"] = str(hsc_data_dir)
        self.h.config["data_request"] = {
            "train": {
                "data": {
                    "dataset_class": "HSCDataSet",
                    "data_location": str(hsc_data_dir),
                    "fields": ["image"],
                }
            },
        }
        self.h.config["data_set"]["use_cache"] = True
        self.h.config["data_set"]["preload_cache"] = False
        self.data_provider = self.h.prepare()["train"]

    def setup(self):
        """
        Prepare for benchmark by defining and setting up the same dataset.
        Despite calling setup_cache this should not trigger another HSC1k download.
        """
        self.setup_cache()
        self.h.config["data_set"]["preload_cache"] = True

    def time_preload_cache_hsc1k(self):
        """Benchmark the amount of time needed to preload the cache of all data"""
        try:
            from hyrax.data_sets.data_cache import DataCache
        except ImportError as e:
            raise NotImplementedError("No DataCache in this version") from e
        self.data_cache = DataCache(self.h.config, self.data_provider)
        self.data_cache.start_preload_thread()
        self.data_cache._preload_thread.join()

    def track_cache_hsc1k_hyrax_size_undercount(self):
        """Benchmark the amount of memory needed to preload the cache with HSC1k data"""
        initial = psutil.Process(os.getpid()).memory_info().rss

        self.time_preload_cache_hsc1k()

        final = psutil.Process(os.getpid()).memory_info().rss
        os_size = final - initial
        hyrax_size = self.data_cache._data_size_bytes
        hyrax_undercount = os_size - hyrax_size
        return (hyrax_undercount / os_size) * 100

    track_cache_hsc1k_hyrax_size_undercount.units = "percent"
