import functools
import logging
from pathlib import Path

import torch
from astropy.table import Table
from tqdm import tqdm

from .lsst_dataset import LSSTDataset

logger = logging.getLogger(__name__)


class DownloadedLSSTDataset(LSSTDataset):
    """
    DownloadedLSSTDataset: A dataset that inherits from LSSTDataset and
    downloads cutouts from the LSST butler and saves them as `.pt` files
    during the first access.On subsequent accesses, it loads the cutouts 
    directly from these files.
    """

    def __init__(self, config):

        self.download_dir = Path(config["general"]["data_dir"])
        self.download_dir.mkdir(exist_ok=True)

        super().__init__(config)

        # Store config for thread-local Butler creation
        self._butler_config = {
            'repo': config["data_set"]["butler_repo"],
            'collections': config["data_set"]["butler_collection"],
            'skymap': config["data_set"]["skymap"]
        }

        # Determine naming strategy
        self._setup_naming_strategy()

    def _setup_naming_strategy(self):
        """Setup file naming strategy based on catalog columns."""
        catalog_columns = self.catalog.colnames if hasattr(self.catalog, 'colnames') else self.catalog.columns

        self.use_object_id = False
        if 'object_id' in catalog_columns:
            self.use_object_id = True
            self.object_id_column = 'object_id'
        elif 'objectId' in catalog_columns:
            self.use_object_id = True
            self.object_id_column = 'objectId'
        else:
            self.object_id_column = 'objectId'

        if not self.use_object_id:
            dataset_length = len(self.catalog)
            self.padding_length = max(4, len(str(dataset_length)))

    def _get_cutout_path(self, idx):
        """Generate cutout file path for a given index."""
        if self.use_object_id:
            if isinstance(self.catalog, Table):
                object_id = self.catalog[idx][self.object_id_column]
            else:
                object_id = self.catalog.iloc[idx][self.object_id_column]
            return self.download_dir / f"cutout_{object_id}.pt"
        else:
            return self.download_dir / f"cutout_{idx:0{self.padding_length}d}.pt"


    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _request_patch_cached(tract_index, patch_index, butler_repo, butler_collections, skymap_name, bands_tuple):
        """
        Cached patch fetching using static method.

        Static method means no 'self' in cache key, making it truly global.
        Thread-safe because each call creates its own Butler instance.
        """
        try:
            import lsst.daf.butler as butler

            # Create fresh Butler instance for this call
            thread_butler = butler.Butler(butler_repo, collections=butler_collections)
            #thread_skymap = thread_butler.get("skyMap", {"skymap": skymap_name})

            data = []
            for band in bands_tuple:  # bands_tuple is hashable for cache key
                butler_dict = {
                    "tract": tract_index,
                    "patch": patch_index,
                    "skymap": skymap_name,
                    "band": band,
                }
                image = thread_butler.get("deep_coadd", butler_dict)
                data.append(image.getImage())

            logger.debug(f"Fetched patch {tract_index}-{patch_index} from Butler")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch patch {tract_index}-{patch_index}: {e}")
            raise

    def _fetch_single_cutout(self, row, idx=None):
        """Fetch cutout, using saved cutout if available."""
        if idx is not None:
            cutout_path = self._get_cutout_path(idx)
            if cutout_path.exists():
                return torch.load(cutout_path, map_location='cpu', weights_only=True)

        # For main thread, use parent's method (original caching)
        import threading
        if threading.current_thread() is threading.main_thread():
            cutout = super()._fetch_single_cutout(row)
        else:
            # For worker threads, use our cached method
            cutout = self._fetch_cutout_with_cache(row)

        # Save cutout if idx provided
        if idx is not None:
            cutout_path = self._get_cutout_path(idx)
            torch.save(cutout, cutout_path)

        return cutout

    def _fetch_cutout_with_cache(self, row):
        """Generate cutout using cached patch fetching."""
        import numpy as np
        from torch import from_numpy

        # Get tract and patch info (using parent's methods)
        tract_info, patch_info = self._get_tract_patch(row)
        box_i = self._parse_box(patch_info, row)

        # Use cached patch fetching - convert bands list to tuple for hashability
        bands_tuple = tuple(self.BANDS)

        patch_images = self._request_patch_cached(
            tract_info.getId(),
            patch_info.sequential_index,
            self._butler_config['repo'],
            self._butler_config['collections'], 
            self._butler_config['skymap'],
            bands_tuple
        )

        # Extract cutout from patch images
        cutout_data = [image[box_i].getArray() for image in patch_images]
        data_np = np.array(cutout_data)
        data_torch = from_numpy(data_np.astype(np.float32))

        return data_torch

    def __getitem__(self, idxs):
        """Modified to pass index for saving cutouts."""
        # Handle single index
        if isinstance(idxs, int):
            row = self.catalog[idxs] if isinstance(self.catalog, Table) else self.catalog.iloc[idxs]
            return self._fetch_single_cutout(row, idx=idxs)

        # Handle multiple indices
        cutouts = []
        for idx in idxs:
            row = self.catalog[idx] if isinstance(self.catalog, Table) else self.catalog.iloc[idx]
            cutouts.append(self._fetch_single_cutout(row, idx=idx))

        return cutouts

    def download_cutouts(self, indices=None):
        """Download cutouts using multiple threads with caching."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if indices is None:
            indices = range(len(self))

        indices_to_download = [idx for idx in indices if not self._get_cutout_path(idx).exists()]

        if not indices_to_download:
            print("All cutouts already downloaded")
            return

        logger.info(f"Downloading {len(indices_to_download)} cutouts using "
                   f"{self._determine_numprocs_download()} threads.")

        with ThreadPoolExecutor(max_workers=self._determine_numprocs_download()) as executor:
            futures = {executor.submit(self._download_single_cutout, idx): idx for idx in indices_to_download}

            with tqdm(total=len(indices_to_download), desc="Downloading cutouts") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        idx = futures[future]
                        logger.error(f"Failed to download cutout {idx}: {e}")
                        pbar.update(1)

        # Log cache stats
        cache_info = self._request_patch_cached.cache_info()
        logger.info(f"Download complete. Cache stats: {cache_info}")

    def _download_single_cutout(self, idx):
        """Helper method to download a single cutout."""
        cutout_path = self._get_cutout_path(idx)
        if cutout_path.exists():
            return

        row = self.catalog[idx] if isinstance(self.catalog, Table) else self.catalog.iloc[idx]
        cutout = self._fetch_cutout_with_cache(row)
        torch.save(cutout, cutout_path)

    def get_cache_info(self):
        """Get cache statistics."""
        return self._request_patch_cached.cache_info()

    def clear_cache(self):
        """Clear the LRU cache."""
        self._request_patch_cached.cache_clear()
        logger.info("Cleared patch cache")

    @staticmethod
    def _determine_numprocs_download():
        """Determine number of threads for downloading."""
        return 1