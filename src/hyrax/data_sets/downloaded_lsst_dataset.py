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

        # Add tracking for band failure statistics
        self._band_failure_stats = {band: 0 for band in self.BANDS}
        self._band_failure_lock = threading.Lock()

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
            required_cols = ["cutout_shape", "filename", "downloaded_bands"]
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
        """Add cutout_shape, filename, and downloaded_bands columns to manifest."""
        if self.catalog_type == "astropy":
            n_rows = len(self.manifest)

            # Create shape column as integer array (assuming 3D tensors like [3, 64, 64])
            empty_shape = np.array([0, 0, 0], dtype=int)  # Placeholder shape
            self.manifest["cutout_shape"] = [empty_shape] * n_rows

            # Create filename column
            self.manifest["filename"] = [""] * n_rows
            self.manifest["filename"] = self.manifest["filename"].astype("U50")  # 50-char strings

            # Add downloaded_bands column to track successful bands in tensor order
            self.manifest["downloaded_bands"] = [""] * n_rows
            self.manifest["downloaded_bands"] = self.manifest["downloaded_bands"].astype("U20")  # e.g., "g,r"
        else:
            # For NestedFrame/HATS, add None columns
            self.manifest["cutout_shape"] = [None] * len(self.manifest)
            self.manifest["filename"] = [None] * len(self.manifest)
            self.manifest["downloaded_bands"] = [None] * len(self.manifest)

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

    def _update_manifest_entry(self, idx, cutout_shape=None, filename="Attempted", downloaded_bands=None):
        """
        Thread-safe manifest update with periodic saves.

        Args:
            idx: Index in the catalog
            cutout_shape: Shape tuple of the cutout tensor, or None for failed downloads
            filename: Basename of the saved file, or "Attempted" only when ALL bands fail
            downloaded_bands: List of band names successfully downloaded in tensor order
        """
        with self._manifest_lock:
            # Update manifest entries
            if cutout_shape is not None:
                shape_array = np.array(list(cutout_shape), dtype=int)
                self.manifest["cutout_shape"][idx] = shape_array
            else:
                # For completely failed downloads
                if self.catalog_type == "astropy":
                    self.manifest["cutout_shape"][idx] = np.array([0, 0, 0], dtype=int)
                else:
                    self.manifest["cutout_shape"][idx] = None

            self.manifest["filename"][idx] = filename

            # Update downloaded_bands tracking in manifest
            if downloaded_bands is not None:
                downloaded_bands_str = ",".join(downloaded_bands) if downloaded_bands else ""
                self.manifest["downloaded_bands"][idx] = downloaded_bands_str
            else:
                self.manifest["downloaded_bands"][idx] = ""

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
                        self._update_manifest_entry(idx, cutout.shape, cutout_path.name, list(self.BANDS))
                        synced_count += 1
                    except Exception as e:
                        logger.warning(f"Could not load existing cutout {cutout_path}: {e}")
            else:
                # File doesn't exist on disk
                if current_filename and current_filename != "Attempted":
                    # Manifest says file exists but it doesn't, reset entry
                    self._update_manifest_entry(idx, None, "", [])
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

            # Track successful data and failed bands separately
            data = []
            failed_bands = []

            for band in bands_tuple:
                butler_dict = {
                    "tract": tract_index,
                    "patch": patch_index,
                    "skymap": skymap_name,
                    "band": band,
                }
                try:
                    image = thread_butler.get("deep_coadd", butler_dict)
                    data.append(image.getImage())
                except Exception as e:
                    logger.warning(f"Failed to fetch band {band} for patch {tract_index}-{patch_index}: {e}")
                    failed_bands.append(band)
                    data.append(None)  # Add None placeholder for failed band; will be filled with NaNs later

            logger.debug(f"Fetched patch {tract_index}-{patch_index} from Butler")
            if failed_bands:
                logger.debug(f"Failed bands for patch {tract_index}-{patch_index}: {failed_bands}")

            return data, failed_bands

        except Exception as e:
            logger.error(f"Failed to fetch patch {tract_index}-{patch_index}: {e}")
            raise

    def _fetch_single_cutout(self, row, idx=None):
        """Fetch cutout, using saved cutout if available."""
        if idx is not None:
            cutout_path = self._get_cutout_path(idx)
            if cutout_path.exists():
                # For cached cutouts, return just the tensor
                return torch.load(cutout_path, map_location="cpu", weights_only=True)

        # For main thread, use parent's method (original caching)
        import threading

        if threading.current_thread() is threading.main_thread():
            cutout = super()._fetch_single_cutout(row)
            downloaded_bands = list(self.BANDS)  # Assume all bands successful with parent method
        else:
            # For worker threads, use our cached method
            # Unpack both cutout and downloaded_bands
            cutout, downloaded_bands = self._fetch_cutout_with_cache(row)

        # Save cutout if idx provided
        if idx is not None:
            cutout_path = self._get_cutout_path(idx)
            torch.save(cutout, cutout_path)

            # Determine if this is a complete failure (all bands failed)
            if len(downloaded_bands) == 0:
                # All bands failed - mark as "Attempted"
                self._update_manifest_entry(idx, None, "Attempted", downloaded_bands)
            else:
                # At least some bands succeeded - save with proper filename
                filename = cutout_path.name
                self._update_manifest_entry(idx, cutout.shape, filename, downloaded_bands)

        return cutout

    def _fetch_cutout_with_cache(self, row):
        """Generate cutout using cached patch fetching with NaN filling for failed bands."""
        from torch import from_numpy

        # Get tract and patch info (using parent's methods)
        tract_info, patch_info = self._get_tract_patch(row)
        box_i = self._parse_box(patch_info, row)

        # Use cached patch fetching - convert bands list to tuple for hashability
        bands_tuple = tuple(self.BANDS)

        # Get patch data and failed bands info
        patch_images, failed_bands = self._request_patch_cached(
            tract_info.getId(),
            patch_info.sequential_index,
            self._butler_config["repo"],
            self._butler_config["collections"],
            self._butler_config["skymap"],
            bands_tuple,
        )

        # Extract cutout with NaN filling for failed bands
        cutout_data = []
        downloaded_bands = []  # Track successfully downloaded bands in order

        for _i, (band, image) in enumerate(zip(self.BANDS, patch_images)):
            if image is not None:
                # Successfully retrieved band
                cutout_data.append(image[box_i].getArray())
                downloaded_bands.append(band)
            else:
                # Failed band - create NaN-filled array with same shape as box
                nan_array = np.full((box_i.getHeight(), box_i.getWidth()), np.nan, dtype=np.float32)
                cutout_data.append(nan_array)
                logger.debug(f"Filled band {band} with NaN for failed retrieval")

        # Update global band failure statistics
        if failed_bands:
            with self._band_failure_lock:
                for band in failed_bands:
                    self._band_failure_stats[band] += 1

        data_np = np.array(cutout_data)
        data_torch = from_numpy(data_np.astype(np.float32))

        # Return cutout and downloaded bands info for manifest tracking
        return data_torch, downloaded_bands

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
                        self._update_manifest_entry(idx, None, "Attempted", [])
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
            cutout, downloaded_bands = self._fetch_cutout_with_cache(row)

            #  Only save cutout and mark as successful if at least one band worked
            if len(downloaded_bands) == 0:
                # All bands failed - don't save file, mark as "Attempted"
                self._update_manifest_entry(idx, None, "Attempted", downloaded_bands)
                raise RuntimeError(f"All bands failed for cutout {idx}")
            else:
                # At least some bands succeeded - save the cutout
                torch.save(cutout, cutout_path)
                filename = cutout_path.name
                self._update_manifest_entry(idx, cutout.shape, filename, downloaded_bands)

        except Exception as e:
            logger.error(f"Failed to download cutout {idx}: {e}")
            # Update manifest with failed attempt (all bands failed)
            self._update_manifest_entry(idx, None, "Attempted", [])
            raise

    def get_cache_info(self):
        """Get cache statistics."""
        return self._request_patch_cached.cache_info()

    def clear_cache(self):
        """Clear the LRU cache."""
        self._request_patch_cached.cache_clear()
        logger.info("Cleared patch cache")

    def get_manifest_stats(self):
        """Get manifest statistics including downloaded bands information."""
        with self._manifest_lock:
            if self.catalog_type == "astropy":
                successful = sum(
                    1 for filename in self.manifest["filename"] if filename and filename != "Attempted"
                )
                failed = sum(1 for filename in self.manifest["filename"] if filename == "Attempted")
                pending = sum(1 for filename in self.manifest["filename"] if not filename)

                # Add statistics about partial downloads (cutouts with missing bands)
                partial_downloads = sum(
                    1
                    for i, (filename, downloaded_bands) in enumerate(
                        zip(self.manifest["filename"], self.manifest["downloaded_bands"])
                    )
                    if filename
                    and filename != "Attempted"
                    and downloaded_bands
                    and len(downloaded_bands.split(",")) < len(self.BANDS)
                )
            else:
                successful = sum(
                    1 for filename in self.manifest["filename"] if filename and filename != "Attempted"
                )
                failed = sum(1 for filename in self.manifest["filename"] if filename == "Attempted")
                pending = sum(1 for filename in self.manifest["filename"] if filename is None)

                partial_downloads = sum(
                    1
                    for i, (filename, downloaded_bands) in enumerate(
                        zip(self.manifest["filename"], self.manifest["downloaded_bands"])
                    )
                    if filename
                    and filename != "Attempted"
                    and downloaded_bands
                    and len(str(downloaded_bands).split(",")) < len(self.BANDS)
                )

            # Collect band failure statistics
            with self._band_failure_lock:
                band_stats = dict(self._band_failure_stats)

            return {
                "total": len(self.manifest),
                "successful": successful,
                "failed": failed,  # All bands failed
                "pending": pending,
                "partial_downloads": partial_downloads,  # Some bands missing
                "band_failure_counts": band_stats,
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

    def get_download_summary(self):
        """
        Get detailed download and band analysis.
        """
        stats = self.get_manifest_stats()

        # Analyze downloaded bands per cutout
        band_success_analysis = {band: 0 for band in self.BANDS}
        complete_downloads = 0

        if self.catalog_type == "astropy":
            downloaded_bands_entries = self.manifest["downloaded_bands"]
        else:
            downloaded_bands_entries = self.manifest["downloaded_bands"]

        for downloaded_bands_str in downloaded_bands_entries:
            if downloaded_bands_str and str(downloaded_bands_str).strip():
                downloaded_bands = [b.strip() for b in str(downloaded_bands_str).split(",") if b.strip()]

                # Count successful downloads per band
                for band in downloaded_bands:
                    if band in band_success_analysis:
                        band_success_analysis[band] += 1

                # Count complete downloads (all bands present)
                if len(downloaded_bands) == len(self.BANDS):
                    complete_downloads += 1

        return {
            "total_cutouts": stats["total"],
            "complete_downloads": complete_downloads,  # All bands present
            "partial_downloads": stats["partial_downloads"],  # Some bands missing
            "failed_downloads": stats["failed"],  # All bands failed
            "pending_downloads": stats["pending"],
            "band_success_counts": band_success_analysis,  # How many cutouts have each band
            "band_failure_counts": stats["band_failure_counts"],  # How many times each band failed
            "expected_bands": list(self.BANDS),
            "percentage_complete": round(complete_downloads / stats["total"] * 100, 2)
            if stats["total"] > 0
            else 0,
        }
