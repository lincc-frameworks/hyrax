from pathlib import Path

import pooch

from hyrax import Hyrax

HSC1K_EXTRACTED_DIRNAME = "hsc_8asec_1000"
HSC1K_ARCHIVE_URL = "doi:10.5281/zenodo.14498536/hsc_demo_data.zip"
HSC1K_ARCHIVE_HASH = "md5:1be05a6b49505054de441a7262a09671"


class DataCacheBenchmarks:
    """Timing benchmarks for per-dataset caching with the HSC1k dataset."""

    def setup_cache(self):
        """Download the HSC1k dataset only once."""
        self.h = Hyrax()

        # asv caches the cwd for each benchmark run
        data_dir = Path("./hsc1k").resolve()
        data_dir.mkdir(exist_ok=True)
        hsc_data_dir = data_dir / HSC1K_EXTRACTED_DIRNAME
        if not hsc_data_dir.exists():
            pooch.retrieve(
                url=HSC1K_ARCHIVE_URL,
                known_hash=HSC1K_ARCHIVE_HASH,
                fname="hsc_demo_data.zip",
                path=data_dir,
                processor=pooch.Unzip(extract_dir="."),
            )

        self.h.config["general"]["results_dir"] = str(data_dir)
        self.h.config["data_request"] = {
            "train": {
                "data": {
                    "dataset_class": "HSCDataset",
                    "data_location": str(hsc_data_dir),
                    "fields": ["image"],
                    "primary_id_field": "object_id",
                }
            },
        }
        self.h.config["data_set"]["use_cache"] = True
        self.data_provider = self.h.prepare()["train"]

    def setup(self):
        """Prepare for benchmark by setting up the dataset."""
        self.setup_cache()

    def time_cache_fill_hsc1k(self):
        """Benchmark the time to fill the cache through normal access."""
        for i in range(len(self.data_provider)):
            self.data_provider[i]

    def time_cache_hit_hsc1k(self):
        """Benchmark cache-hit performance after the cache is filled."""
        for i in range(len(self.data_provider)):
            self.data_provider[i]
        # Now measure cache hit
        self.data_provider[0]
