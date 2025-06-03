import functools
import logging
import threading
from pathlib import Path

import numpy as np
import torch
from astropy.table import Table
from tqdm import tqdm

from .lsst_dataset import LSSTDataset

logger = logging.getLogger(__name__)


class DownloadedLSSTDataset(LSSTDataset):
    """
    DownloadedLSSTDataset: A dataset that inherits from LSSTDataset and downloads
    cutouts from the LSST butler, saving them as `.pt` files during first access.
    On subsequent accesses, it loads cutouts directly from these cached files.

    This class also creates a manifest files with the shape of each cutout and the
    corresponding filename.

    Public Methods:
        download_cutouts(indices=None, sync_filesystem=True, max_workers=None, force_retry=False):
            Download cutouts with parallel processing. Automatically resumes from
            previous progress. Use max_workers to control thread count, force_retry
            to re-attempt failed downloads.

        get_manifest_stats():
            Returns dict with download statistics: total, successful, failed, pending
            counts and manifest file path.

        get_download_progress():
            Returns detailed progress metrics including completion percentage and
            failure rates.

        reset_failed_downloads():
            Resets all failed download attempts to allow retry without force_retry flag.
            Returns count of reset entries.

        save_manifest_now():
            Forces immediate manifest save (normally saved periodically during downloads).

        get_cache_info():
            Returns LRU cache statistics for patch fetching performance monitoring.

        clear_cache():
            Clears the patch LRU cache to free memory.

    Usage Example:
        # Initialize Hyrax
        h = hyrax.Hyrax()
        a = h.prepare()

        # Download all cutouts (resumes automatically)
        a.download_cutouts(max_workers=4)
        WARNING: The LRU Caching scheme is slightly complicated, so it is recommended to
        use the default max_workers=1 for the first download. Simply using more workers
        may not always speed up the download process.

        # Check progress
        a.get_download_progress()

        # Retry failed downloads
        a.download_cutouts(force_retry=True)

        # Access cutouts (loads from cache)
        cutout = a[0]  # Single cutout
        cutouts = a[0:10]  # Multiple cutouts

    File Organization:
    - Cutouts saved as: cutout_{object_id}.pt or cutout_{index:04d}.pt
    - Manifest saved as: manifest.fits (Astropy) or manifest.parquet (HATS)
    - All files stored in config["general"]["data_dir"]
    """

    def __init__(self, config):
        self.download_dir = Path(config["general"]["data_dir"])
        self.download_dir.mkdir(exist_ok=True)

        # Preventing name collision with parent class config
        self._config = config

        # Initialize parent class with config
        super().__init__(config)

        # Store config for thread-local Butler creation
        self._butler_config = {
            "repo": config["data_set"]["butler_repo"],
            "collections": config["data_set"]["butler_collection"],
            "skymap": config["data_set"]["skymap"],
        }

        # Manifest management
        self._manifest_lock = threading.Lock()
        self._updates_since_save = 0
        self._save_interval = 1000

        # Determine naming strategy and initialize manifest
        self._setup_naming_strategy()
        self._initialize_manifest()

    def _setup_naming_strategy(self):
        """Setup file naming strategy based on catalog columns."""
        catalog_columns = self.catalog.colnames if hasattr(self.catalog, "colnames") else self.catalog.columns

        self.use_object_id = False
        if self._config["data_set"]["object_id_column_name"]:
            self.use_object_id = True
            self.object_id_column = self._config["data_set"]["object_id_column_name"]
        elif "object_id" in catalog_columns:
            self.use_object_id = True
            self.object_id_column = "object_id"
        elif "objectId" in catalog_columns:
            self.use_object_id = True
            self.object_id_column = "objectId"
        else:
            self.object_id_column = "objectId"

        if not self.use_object_id:
            dataset_length = len(self.catalog)
            self.padding_length = max(4, len(str(dataset_length)))

    def _initialize_manifest(self):
        """Create or load existing manifest with cutout tracking columns."""

        if isinstance(self.catalog, Table):
            self.catalog_type = "astropy"
            self.manifest_path = self.download_dir / "manifest.fits"
        else:
            self.catalog_type = "hats"
            self.manifest_path = self.download_dir / "manifest.parquet"

        # Try to load existing manifest first
        if self.manifest_path.exists():
            logger.info(f"Loading existing manifest from {self.manifest_path}")
            try:
                self.manifest = self._load_existing_manifest()

                # Validate manifest compatibility with current catalog
                if self._validate_manifest_compatibility():
                    logger.info("Existing manifest is compatible with current catalog.")
                    return
                else:
                    logger.warning("Manifest incompatible with current catalog, creating new one")
                    # Continue to create new manifest
            except Exception as e:
                logger.warning(f"Failed to load existing manifest: {e}, creating new one")
                # Continue to create new manifest

        # Create new manifest
        logger.info("Creating new manifest")
        if self.catalog_type == "astropy":
            self.manifest = Table()
            # Copy only the data columns
            for col_name in self.catalog.colnames:
                self.manifest[col_name] = self.catalog[col_name]
        else:
            self.manifest = self.catalog.copy()

        self._add_manifest_columns()
        self._save_manifest()
        logger.info(f"Initialized new manifest at {self.manifest_path}")

    def _load_existing_manifest(self):
        """Load existing manifest file."""
        if self.catalog_type == "astropy":
            return Table.read(self.manifest_path)
        else:
            import pandas as pd

            df = pd.read_parquet(self.manifest_path)
            # Convert back to original catalog type if needed
            return df

    def _validate_manifest_compatibility(self):
        """Check if existing manifest is compatible with current catalog."""
        try:
            # Check if lengths match
            if len(self.manifest) != len(self.catalog):
                logger.warning(
                    f"Manifest length ({len(self.manifest)}) != catalog\
                        length ({len(self.catalog)})"
                )
                return False

            # Check if required columns exist
            required_cols = ["cutout_shape", "filename"]
            if self.catalog_type == "astropy":
                manifest_cols = self.manifest.colnames
            else:
                manifest_cols = self.manifest.columns

            for col in required_cols:
                if col not in manifest_cols:
                    logger.warning(f"Required column '{col}' missing from manifest")
                    return False

            # Check if key identifying columns match (basic validation)
            if self.use_object_id and self.object_id_column in manifest_cols:
                # Compare a few object IDs to ensure consistency
                sample_size = min(10, len(self.catalog))
                for i in range(sample_size):
                    if self.catalog_type == "astropy":
                        cat_id = self.catalog[i][self.object_id_column]
                        man_id = self.manifest[i][self.object_id_column]
                    else:
                        cat_id = self.catalog.iloc[i][self.object_id_column]
                        man_id = self.manifest.iloc[i][self.object_id_column]

                    if cat_id != man_id:
                        logger.warning(f"Object ID mismatch at index {i}: {cat_id} != {man_id}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Error validating manifest compatibility: {e}")
            return False

    def _add_manifest_columns(self):
        """Add cutout_shape and filename columns to manifest."""
        if self.catalog_type == "astropy":
            n_rows = len(self.manifest)

            # Create shape column as integer array (assuming 3D tensors like [3, 64, 64])
            empty_shape = np.array([0, 0, 0], dtype=int)  # Placeholder shape
            self.manifest["cutout_shape"] = [empty_shape] * n_rows

            # Create filename column
            self.manifest["filename"] = [""] * n_rows
            self.manifest["filename"] = self.manifest["filename"].astype("U50")  # 50-char strings
        else:
            # For NestedFrame/HATS, add None columns
            self.manifest["cutout_shape"] = [None] * len(self.manifest)
            self.manifest["filename"] = [None] * len(self.manifest)

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

    def _update_manifest_entry(self, idx, cutout_shape=None, filename="Attempted"):
        """
        Thread-safe manifest update with periodic saves.

        Args:
            idx: Index in the catalog
            cutout_shape: Shape tuple of the cutout tensor, or None for failed downloads
            filename: Basename of the saved file, or "Attempted" for failures
        """
        with self._manifest_lock:
            # Update manifest entries
            if cutout_shape is not None:
                shape_array = np.array(list(cutout_shape), dtype=int)
                self.manifest["cutout_shape"][idx] = shape_array
            else:
                # For failed downloads
                if self.catalog_type == "astropy":
                    self.manifest["cutout_shape"][idx] = np.array([0, 0, 0], dtype=int)
                else:
                    self.manifest["cutout_shape"][idx] = None

            self.manifest["filename"][idx] = filename

            # Increment update counter and save periodically
            self._updates_since_save += 1
            if self._updates_since_save >= self._save_interval:
                self._save_manifest()
                self._updates_since_save = 0
                logger.debug(f"Periodic manifest save completed ({self._save_interval} updates)")

    def _save_manifest(self):
        """Save manifest in appropriate format (FITS for Astropy, Parquet for HATS)."""
        try:
            if self.catalog_type == "astropy":
                self.manifest.write(self.manifest_path, overwrite=True)
            else:
                # For HATS catalogs, save as Parquet
                # Convert to pandas DataFrame if needed
                manifest_df = self.manifest.compute() if hasattr(self.manifest, "compute") else self.manifest
                manifest_df.to_parquet(self.manifest_path)

            logger.debug(f"Manifest saved to {self.manifest_path}")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def _sync_manifest_with_filesystem(self):
        """Sync manifest with actual downloaded files on disk."""
        logger.info("Syncing manifest with filesystem...")
        synced_count = 0

        for idx in range(len(self.manifest)):
            cutout_path = self._get_cutout_path(idx)

            # Get current manifest state
            if self.catalog_type == "astropy":
                current_filename = self.manifest["filename"][idx]
            else:
                current_filename = self.manifest.iloc[idx]["filename"]

            if cutout_path.exists():
                # File exists on disk
                if not current_filename or current_filename == "Attempted":
                    # Manifest doesn't reflect the file exists, update it
                    try:
                        cutout = torch.load(cutout_path, map_location="cpu", weights_only=True)
                        self._update_manifest_entry(idx, cutout.shape, cutout_path.name)
                        synced_count += 1
                    except Exception as e:
                        logger.warning(f"Could not load existing cutout {cutout_path}: {e}")
            else:
                # File doesn't exist on disk
                if current_filename and current_filename != "Attempted":
                    # Manifest says file exists but it doesn't, reset entry
                    self._update_manifest_entry(idx, None, "")
                    synced_count += 1

        if synced_count > 0:
            logger.info(f"Synced {synced_count} manifest entries with filesystem")
            self.save_manifest_now()

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _request_patch_cached(
        tract_index, patch_index, butler_repo, butler_collections, skymap_name, bands_tuple
    ):
        """
        Cached patch fetching using static method.

        Static method means no 'self' in cache key, making it truly global.
        Thread-safe because each call creates its own Butler instance.
        """
        try:
            import lsst.daf.butler as butler

            # Create fresh Butler instance for this call
            thread_butler = butler.Butler(butler_repo, collections=butler_collections)

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
                return torch.load(cutout_path, map_location="cpu", weights_only=True)

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

            # Update manifest with successful download
            filename = cutout_path.name  # Just the basename
            self._update_manifest_entry(idx, cutout.shape, filename)

        return cutout

    def _fetch_cutout_with_cache(self, row):
        """Generate cutout using cached patch fetching."""
        from torch import from_numpy

        # Get tract and patch info (using parent's methods)
        tract_info, patch_info = self._get_tract_patch(row)
        box_i = self._parse_box(patch_info, row)

        # Use cached patch fetching - convert bands list to tuple for hashability
        bands_tuple = tuple(self.BANDS)

        patch_images = self._request_patch_cached(
            tract_info.getId(),
            patch_info.sequential_index,
            self._butler_config["repo"],
            self._butler_config["collections"],
            self._butler_config["skymap"],
            bands_tuple,
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

    def download_cutouts(self, indices=None, sync_filesystem=True, max_workers=None, force_retry=False):
        """Download cutouts using multiple threads with caching.

        Args:
            indices: List of indices to download, or None for all
            sync_filesystem: Whether to sync manifest with existing files on disk
            max_workers: Maximum number of worker threads, or None to use default
            force_retry: Whether to retry previously failed downloads
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if indices is None:
            indices = range(len(self))

        # Optionally sync manifest with filesystem before downloading
        if sync_filesystem:
            self._sync_manifest_with_filesystem()

        # Determine which cutouts need downloading
        indices_to_download = []
        for idx in indices:
            cutout_path = self._get_cutout_path(idx)

            # Check if file exists on disk
            if cutout_path.exists():
                continue

            # Check manifest status
            if self.catalog_type == "astropy":
                filename = self.manifest["filename"][idx]
            else:
                filename = self.manifest.iloc[idx]["filename"]

            # Skip if already attempted and failed (unless force_retry is True)
            if filename == "Attempted" and not force_retry:
                logger.debug(f"Skipping previously failed download for index {idx}")
                continue

            indices_to_download.append(idx)

        if not indices_to_download:
            logger.info("All cutouts already downloaded")
            return

        # Determine number of workers
        if max_workers is None:
            max_workers = self._determine_numprocs_download()

        logger.info(f"Downloading {len(indices_to_download)} cutouts using {max_workers} threads.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._download_single_cutout, idx): idx for idx in indices_to_download}

            with tqdm(total=len(indices_to_download), desc="Downloading cutouts") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        idx = futures[future]
                        logger.error(f"Failed to download cutout {idx}: {e}")
                        self._update_manifest_entry(idx, None, "Attempted")
                        pbar.update(1)

        # Final manifest save
        with self._manifest_lock:
            if self._updates_since_save > 0:
                self._save_manifest()
                self._updates_since_save = 0

        # Log cache and download stats
        cache_info = self._request_patch_cached.cache_info()
        logger.info(f"Download complete. Cache stats: {cache_info}")
        logger.info(f"Manifest saved to {self.manifest_path}")

    def _download_single_cutout(self, idx):
        """Helper method to download a single cutout."""
        cutout_path = self._get_cutout_path(idx)
        if cutout_path.exists():
            return

        try:
            row = self.catalog[idx] if isinstance(self.catalog, Table) else self.catalog.iloc[idx]
            cutout = self._fetch_cutout_with_cache(row)
            torch.save(cutout, cutout_path)

            # Update manifest with successful download
            filename = cutout_path.name  # Just the basename
            self._update_manifest_entry(idx, cutout.shape, filename)

        except Exception as e:
            logger.error(f"Failed to download cutout {idx}: {e}")
            # Update manifest with failed attempt
            self._update_manifest_entry(idx, None, "Attempted")
            raise

    def get_cache_info(self):
        """Get cache statistics."""
        return self._request_patch_cached.cache_info()

    def clear_cache(self):
        """Clear the LRU cache."""
        self._request_patch_cached.cache_clear()
        logger.info("Cleared patch cache")

    def get_manifest_stats(self):
        """Convnience function to get manifest statistics.

        # To use:
        h = hyrax.Hyrax(config)
        a = h.prepare()
        a.download_cutouts()
        a.get_manifest_stats()
        """
        with self._manifest_lock:
            if self.catalog_type == "astropy":
                successful = sum(
                    1 for filename in self.manifest["filename"] if filename and filename != "Attempted"
                )
                failed = sum(1 for filename in self.manifest["filename"] if filename == "Attempted")
                pending = sum(1 for filename in self.manifest["filename"] if not filename)
            else:
                successful = sum(
                    1 for filename in self.manifest["filename"] if filename and filename != "Attempted"
                )
                failed = sum(1 for filename in self.manifest["filename"] if filename == "Attempted")
                pending = sum(1 for filename in self.manifest["filename"] if filename is None)

            return {
                "total": len(self.manifest),
                "successful": successful,
                "failed": failed,
                "pending": pending,
                "manifest_path": str(self.manifest_path),
            }

    def save_manifest_now(self):
        """Force immediate manifest save."""
        with self._manifest_lock:
            self._save_manifest()
            self._updates_since_save = 0
        logger.info("Manifest manually saved")

    @staticmethod
    def _determine_numprocs_download():
        """Determine number of threads for downloading."""
        # TODO:This is a placeholder for actual logic to determine number of threads.
        return 1

    def reset_failed_downloads(self):
        """Reset failed download attempts to allow retry."""
        reset_count = 0

        for idx in range(len(self.manifest)):
            if self.catalog_type == "astropy":
                filename = self.manifest["filename"][idx]
            else:
                filename = self.manifest.iloc[idx]["filename"]

            if filename == "Attempted":
                self._update_manifest_entry(idx, None, "")
                reset_count += 1

        if reset_count > 0:
            logger.info(f"Reset {reset_count} failed download attempts")
            self.save_manifest_now()

        return reset_count

    def get_download_progress(self):
        """Get detailed download progress information."""
        stats = self.get_manifest_stats()

        # Calculate additional metrics
        total = stats["total"]
        successful = stats["successful"]
        failed = stats["failed"]
        pending = stats["pending"]

        progress_percent = (successful / total * 100) if total > 0 else 0
        failure_rate = (failed / (successful + failed) * 100) if (successful + failed) > 0 else 0

        return {
            **stats,
            "progress_percent": round(progress_percent, 2),
            "failure_rate": round(failure_rate, 2),
            "completed": successful + failed,
            "remaining": pending,
        }
