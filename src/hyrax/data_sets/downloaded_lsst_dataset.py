import functools
import logging
import threading
from pathlib import Path

import numpy as np
import torch
from astropy.table import Table
from tqdm import tqdm

from .lsst_dataset import LSSTDataset
from .tensor_cache_mixin import TensorCacheMixin

logger = logging.getLogger(__name__)


class DownloadedLSSTDataset(LSSTDataset, TensorCacheMixin):
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

        manifest_stats():
            Returns dict with download statistics: total, successful, failed, pending
            counts and manifest file path.

        download_progress():
            Returns detailed progress metrics including completion percentage and
            failure rates.

        reset_failed_downloads():
            Resets all failed download attempts to allow retry without force_retry flag.
            Returns count of reset entries.

        save_manifest_now():
            Forces immediate manifest save (normally saved periodically during downloads).

        cache_info():
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
        a.download_progress()

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

    def __init__(self, config, data_location):
        self.download_dir = Path(data_location)
        self.download_dir.mkdir(exist_ok=True)

        # Preventing name collision with parent class config
        self._config = config

        # Initialize parent class with config
        super().__init__(config, data_location=data_location)

        try:
            import lsst.daf.butler as _butler  # noqa: F401

            self._butler_config = {
                "repo": config["data_set"]["butler_repo"],
                "collections": config["data_set"]["butler_collection"],
                "skymap": config["data_set"]["skymap"],
            }
        except ImportError:
            self._butler_config = None

        # Manifest management
        self._manifest_lock = threading.Lock()
        self._updates_since_save = 0
        self._save_interval = 1000

        # Determine naming strategy and initialize manifest (includes band filtering validation)
        self._setup_naming_strategy()
        self._initialize_manifest()

        # Add tracking for band failure statistics (use current BANDS which may be filtered)
        self._band_failure_stats = {band: 0 for band in self.BANDS}
        self._band_failure_lock = threading.Lock()

        # Initialize filtering state
        self._manifest_filter_object_ids = None
        self._catalog_to_manifest_index_map = None
        self._manifest_to_catalog_index_map = None

        # Initialize tensor caching from mixin
        self._init_tensor_cache(config)

    def get_objectId(self, idx):  # noqa: N802
        """Get object ID for a given index based on naming strategy."""
        if self.use_object_id:
            if isinstance(self.catalog, Table):
                return str(self.catalog[idx][self.object_id_column])
            else:
                return str(self.catalog.iloc[idx][self.object_id_column])
        else:
            return str(idx)

    def ids(self, log_every=None):
        """Generator yielding object IDs for the entire dataset. Required by TensorCacheMixin"""
        log = log_every is not None and isinstance(log_every, int)

        for idx in range(len(self.catalog)):
            if log and idx != 0 and idx % log_every == 0:
                logger.info(f"Processed {idx} objects")
            yield self.get_objectId(idx)

        # Final log message after completing iteration
        if log and len(self.catalog) > 0:
            logger.info(f"Processed {len(self.catalog)} objects")

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
        """Create new manifest or load/merge with existing manifest, with band filtering validation."""

        if isinstance(self.catalog, Table):
            self.catalog_type = "astropy"
            self.manifest_path = self.download_dir / "manifest.fits"
        else:
            self.catalog_type = "hats"
            self.manifest_path = self.download_dir / "manifest.parquet"

        # Initialize band filtering flags
        self._is_filtering_bands = False
        self._original_bands = None
        self._filtered_bands = None
        self._band_indices = None

        # Try to load existing manifest first
        if not self.manifest_path.exists():
            if self._butler_config is None:
                msg = "Cannot find any data source. There is no existing manifest, and there is no "
                msg += "butler available. Please try to run this on an RSP where a butler is available or "
                msg += "ensure a proper manifest and cutouts are available in "
                msg += f"{self.config['general']['data_dir']}"
                raise RuntimeError(msg)

            # Create new manifest (no existing manifest found)
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
        else:
            logger.info(f"Found existing manifest at {self.manifest_path}")
            try:
                existing_manifest = self._load_existing_manifest()

                # Check for band filtering opportunity
                available_bands_set, original_band_order = self._get_available_bands_from_manifest(
                    existing_manifest
                )

                if available_bands_set is not None and original_band_order is not None:
                    requested_bands = set(self.BANDS)

                    # Only setup filtering if requested bands are a PROPER SUBSET
                    if requested_bands < available_bands_set:  # Proper subset (not equal)
                        logger.info(
                            f"Requested bands {sorted(list(requested_bands))} are a subset of "
                            f"available {sorted(list(available_bands_set))}"
                        )
                        self._setup_band_filtering(requested_bands, original_band_order)
                    elif requested_bands == available_bands_set:
                        logger.info("Requested bands match available bands exactly, no filtering needed")
                    else:
                        missing_bands = requested_bands - available_bands_set
                        raise ValueError(
                            f"Requested bands {sorted(list(missing_bands))} are not available in downloads. "
                            f"Available bands: {sorted(list(available_bands_set))}. "
                            f"Please set up a new data directory or download missing bands first."
                        )

                # Perform manifest merge
                self.manifest, merge_stats = self._merge_manifests(existing_manifest)

                # Log merge results
                logger.info(
                    f"Manifest merge completed: {merge_stats['preserved']} preserved, "
                    f"{merge_stats['added']} added"
                )

                # Warn about new objects that need downloading
                if merge_stats["added"] > 0:
                    logger.warning(
                        f"{merge_stats['added']} new objects were added to the manifest "
                        f"but are not yet downloaded. Consider running download_cutouts() to "
                        f"download these missing objects."
                    )

                # Save the merged manifest
                self._save_manifest()

            except Exception as e:
                logger.error(f"Failed to load/merge existing manifest: {e}")
                logger.error(
                    "Cannot proceed with incompatible manifest. Specify new data directory to continue."
                )
                raise

    def _load_existing_manifest(self):
        """Load existing manifest file."""
        if self.catalog_type == "astropy":
            return Table.read(self.manifest_path)
        else:
            import pandas as pd

            df = pd.read_parquet(self.manifest_path)
            # Convert back to original catalog type if needed
            return df

    def _merge_manifests(self, existing_manifest):
        """Merge existing manifest with current catalog based on object_id."""

        # Verify object_id merging is possible
        if not self.use_object_id:
            raise ValueError("Cannot merge manifests without object_id")

        # Check required columns exist in existing manifest
        existing_cols = (
            existing_manifest.colnames if self.catalog_type == "astropy" else existing_manifest.columns
        )
        required_cols = ["cutout_shape", "filename", "downloaded_bands"]

        for col in required_cols:
            if col not in existing_cols:
                raise ValueError(f"Existing manifest missing required column: {col}")

        # Create object_id sets for comparison
        if self.catalog_type == "astropy":
            current_object_ids = set(self.catalog[self.object_id_column])
            existing_object_ids = set(existing_manifest[self.object_id_column])
        else:
            current_object_ids = set(self.catalog[self.object_id_column])
            existing_object_ids = set(existing_manifest[self.object_id_column])

        # Check if current catalog is a subset of existing manifest
        if current_object_ids.issubset(existing_object_ids):
            # All objects exist in manifest - just filter for current operations
            logger.info(
                f"Current catalog ({len(current_object_ids)} objects)\
                            is a subset of existing manifest "
                f"({len(existing_object_ids)} objects). Using existing manifest\
                            with filtering for operations."
            )

            # Keep the FULL existing manifest but store filtering info for operations
            self._manifest_filter_object_ids = current_object_ids
            self._build_catalog_to_manifest_index_map(existing_manifest)
            merge_stats = {
                "preserved": len(existing_object_ids),
                "added": 0,
                "filtered_for_operations": len(current_object_ids),
            }
            return existing_manifest, merge_stats

        # Current catalog contains new objects - add them to existing manifest
        logger.info(
            f"Current catalog contains {len(current_object_ids - existing_object_ids)} new objects. "
            f"Adding to existing manifest while preserving all {len(existing_object_ids)}\
                                            existing objects."
        )

        # Start with existing manifest to preserve all data
        if self.catalog_type == "astropy":
            merged_manifest = existing_manifest.copy()
        else:
            merged_manifest = existing_manifest.copy()

        # Find new objects to add
        new_object_ids = current_object_ids - existing_object_ids
        if new_object_ids:
            # Get rows for new objects from current catalog
            new_rows_indices = []
            for i in range(len(self.catalog)):
                if self.catalog_type == "astropy":
                    obj_id = self.catalog[i][self.object_id_column]
                else:
                    obj_id = self.catalog.iloc[i][self.object_id_column]

                if obj_id in new_object_ids:
                    new_rows_indices.append(i)

            # Extract new rows from catalog
            if self.catalog_type == "astropy":
                new_rows = self.catalog[new_rows_indices]

                # Add manifest columns for new rows
                n_new_rows = len(new_rows)
                empty_shape = np.array([0, 0, 0], dtype=int)
                new_rows["cutout_shape"] = [empty_shape] * n_new_rows
                new_rows["filename"] = [""] * n_new_rows
                new_rows["filename"] = new_rows["filename"].astype("U50")
                new_rows["downloaded_bands"] = [""] * n_new_rows
                new_rows["downloaded_bands"] = new_rows["downloaded_bands"].astype("U20")

                # Stack the tables
                from astropy.table import vstack

                merged_manifest = vstack([merged_manifest, new_rows])
            else:
                new_rows = self.catalog.iloc[new_rows_indices].copy()

                # Add manifest columns for new rows
                new_rows["cutout_shape"] = [None] * len(new_rows)
                new_rows["filename"] = [None] * len(new_rows)
                new_rows["downloaded_bands"] = [None] * len(new_rows)

                # Concatenate dataframes
                import pandas as pd

                merged_manifest = pd.concat([merged_manifest, new_rows], ignore_index=True)

        merge_stats = {
            "preserved": len(existing_object_ids),
            "added": len(new_object_ids),
            "total_in_manifest": len(merged_manifest),
        }

        return merged_manifest, merge_stats

    def _build_catalog_to_manifest_index_map(self, manifest):
        """Build efficient mapping from catalog indices to manifest indices."""
        if self._manifest_filter_object_ids is None:
            # No filtering needed
            self._catalog_to_manifest_index_map = None
            self._manifest_to_catalog_index_map = None
            return

        # Create object_id to manifest index lookup
        manifest_lookup = {}
        for manifest_idx in range(len(manifest)):
            if isinstance(manifest, Table):
                obj_id = manifest[manifest_idx][self.object_id_column]
            else:
                obj_id = manifest.iloc[manifest_idx][self.object_id_column]
            manifest_lookup[obj_id] = manifest_idx

        # Build catalog index to manifest index mapping and reverse mapping
        self._catalog_to_manifest_index_map = {}
        self._manifest_to_catalog_index_map = {}
        for catalog_idx in range(len(self.catalog)):
            if isinstance(self.catalog, Table):
                catalog_obj_id = self.catalog[catalog_idx][self.object_id_column]
            else:
                catalog_obj_id = self.catalog.iloc[catalog_idx][self.object_id_column]

            if catalog_obj_id in manifest_lookup:
                manifest_idx = manifest_lookup[catalog_obj_id]
                self._catalog_to_manifest_index_map[catalog_idx] = manifest_idx
                self._manifest_to_catalog_index_map[manifest_idx] = catalog_idx
            else:
                raise ValueError(f"Object ID {catalog_obj_id} from catalog not found in manifest")

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

    def _get_available_bands_from_manifest(self, manifest):
        """Get available bands by checking first 10 successful downloads for consistency."""
        if len(manifest) == 0:
            return None, None

        successful_entries = []

        # Find first 10 successful downloads (check last up to 12 entries, but handle small manifests)
        start_idx = max(0, len(manifest) - 12)
        for i in range(start_idx, len(manifest)):
            if len(successful_entries) >= 10:
                break

            if self.catalog_type == "astropy":
                filename = manifest["filename"][i]
                downloaded_bands_str = manifest["downloaded_bands"][i]
            else:
                filename = manifest.iloc[i]["filename"]
                downloaded_bands_str = manifest.iloc[i]["downloaded_bands"]

            # Only consider successful downloads
            if (
                filename
                and filename != "Attempted"
                and downloaded_bands_str
                and str(downloaded_bands_str).strip()
            ):
                bands = [b.strip() for b in str(downloaded_bands_str).split(",") if b.strip()]
                if bands:  # Non-empty band list
                    successful_entries.append(bands)

        if not successful_entries:
            return None, None

        # Check that all successful entries have identical band lists
        first_bands = successful_entries[0]
        for i, bands in enumerate(successful_entries[1:], 1):
            if bands != first_bands:
                raise ValueError(
                    f"Inconsistent band ordering in manifest. Entry 0 has {first_bands}, "
                    f"but entry {i} has {bands}. Cannot determine consistent band structure."
                )

        return set(first_bands), first_bands

    def _setup_band_filtering(self, requested_bands, original_band_order):
        """Setup band filtering to extract only requested bands from cached cutouts."""
        # Store filtering info
        self._original_bands = original_band_order
        self._filtered_bands = [band for band in original_band_order if band in requested_bands]
        self._is_filtering_bands = True

        # Create mapping from filtered bands to original tensor indices
        self._band_indices = []
        for band in self._filtered_bands:
            self._band_indices.append(self._original_bands.index(band))

        # Override the BANDS property to reflect filtered bands
        self.BANDS = tuple(self._filtered_bands)

        logger.info(f"Band filtering setup: {self._original_bands} -> {self._filtered_bands}")
        logger.info(f"Tensor indices to extract: {self._band_indices}")

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
            idx: Index in the manifest
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

        # When filtering is active, we need to map manifest indices to catalog indices
        for manifest_idx in range(len(self.manifest)):
            # Find the corresponding catalog index for this manifest entry
            catalog_idx = None
            if self._manifest_to_catalog_index_map is not None:
                # Filtering is active - use reverse mapping for O(1) lookup
                catalog_idx = self._manifest_to_catalog_index_map.get(manifest_idx)
                # If no catalog index maps to this manifest index, skip (object not in current catalog)
                if catalog_idx is None:
                    continue
            else:
                # No filtering - direct mapping
                catalog_idx = manifest_idx
                # Ensure catalog index is within bounds
                if catalog_idx >= len(self.catalog):
                    continue

            cutout_path = self._get_cutout_path(catalog_idx)

            # Get current manifest state
            if self.catalog_type == "astropy":
                current_filename = self.manifest["filename"][manifest_idx]
            else:
                current_filename = self.manifest.iloc[manifest_idx]["filename"]

            if cutout_path.exists():
                # File exists on disk
                if not current_filename or current_filename == "Attempted":
                    # Manifest doesn't reflect the file exists, update it
                    try:
                        cutout = torch.load(cutout_path, map_location="cpu", weights_only=True)
                        bands_for_existing = (
                            list(self._original_bands) if self._is_filtering_bands else list(self.BANDS)
                        )
                        self._update_manifest_entry(
                            manifest_idx, cutout.shape, cutout_path.name, bands_for_existing
                        )
                        synced_count += 1
                    except Exception as e:
                        logger.warning(f"Could not load existing cutout {cutout_path}: {e}")
            else:
                # File doesn't exist on disk
                if current_filename and current_filename != "Attempted":
                    # Manifest says file exists but it doesn't, reset entry
                    self._update_manifest_entry(manifest_idx, None, "", [])
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

    def _fetch_single_cutout(self, row, idx=None, manifest_idx=None):
        """Fetch cutout, using saved cutout if available, with optional band filtering."""
        if idx is not None:
            cutout_path = self._get_cutout_path(idx)
            if cutout_path.exists():
                # Use tensor caching for already downloaded cutouts
                if self.use_object_id:
                    if isinstance(self.catalog, Table):
                        object_id = str(self.catalog[idx][self.object_id_column])
                    else:
                        object_id = str(self.catalog.iloc[idx][self.object_id_column])

                    # Use cached tensor loading from mixin
                    cutout = self._object_id_to_tensor_cached(object_id)
                else:
                    # Fallback for non-object-id based datasets
                    cutout = torch.load(cutout_path, map_location="cpu", weights_only=True)

                    # Apply band filtering if needed
                    if self._is_filtering_bands and self._band_indices is not None:
                        cutout = cutout[self._band_indices]
                        logger.debug(f"Applied band filtering to cached cutout {idx}: {cutout.shape}")

                return self.apply_transform(cutout)

        # For main thread, use parent's method (original caching)
        import threading

        if threading.current_thread() is threading.main_thread():
            cutout = super()._fetch_single_cutout(row)
            # When using parent method, assume all original bands were successful
            downloaded_bands = list(self._original_bands) if self._is_filtering_bands else list(self.BANDS)
        else:
            # For worker threads, use our cached method
            cutout, downloaded_bands = self._fetch_cutout_with_cache(row)

        # Apply band filtering to new downloads if needed
        original_cutout_shape = cutout.shape
        if self._is_filtering_bands and self._band_indices is not None:
            cutout = cutout[self._band_indices]
            # Update downloaded_bands to reflect only the filtered bands that were actually present
            downloaded_bands = []
            for i in self._band_indices:
                if i >= len(self._original_bands):
                    raise ValueError(
                        f"Band index {i} is out of bounds for\
                                original_bands (length {len(self._original_bands)}). "
                        f"This indicates a bug in band filtering setup."
                    )
                downloaded_bands.append(self._original_bands[i])
            logger.debug(f"Applied band filtering to new cutout: {original_cutout_shape} -> {cutout.shape}")

        # Save cutout if idx provided (save the filtered version)
        if idx is not None:
            cutout_path = self._get_cutout_path(idx)
            torch.save(cutout, cutout_path)

            # Use manifest_idx for updating manifest, fallback to idx if not provided
            update_idx = manifest_idx if manifest_idx is not None else idx

            # Determine if this is a complete failure (all bands failed)
            if len(downloaded_bands) == 0:
                # All bands failed - mark as "Attempted"
                self._update_manifest_entry(update_idx, None, "Attempted", downloaded_bands)
            else:
                # At least some bands succeeded - save with proper filename
                filename = cutout_path.name
                self._update_manifest_entry(update_idx, cutout.shape, filename, downloaded_bands)

        return self.apply_transform(cutout)

    def _fetch_cutout_with_cache(self, row):
        """Generate cutout using cached patch fetching with NaN filling for failed bands."""
        from torch import from_numpy

        # Get tract and patch info (using parent's methods)
        tract_info, patch_info = self._get_tract_patch(row)
        box_i = self._parse_box(patch_info, row)

        # Use cached patch fetching - convert bands list to tuple for hashability
        bands_tuple = tuple(self._original_bands) if self._is_filtering_bands else tuple(self.BANDS)

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

        bands_to_process = self._original_bands if self._is_filtering_bands else self.BANDS
        for _i, (band, image) in enumerate(zip(bands_to_process, patch_images)):
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

    def _load_tensor_for_cache(self, object_id: str):
        """Implementation of TensorCacheMixin abstract method."""
        # Find the catalog index for this object_id
        catalog_idx = None

        if isinstance(self.catalog, Table):
            for i in range(len(self.catalog)):
                if str(self.catalog[i][self.object_id_column]) == object_id:
                    catalog_idx = i
                    break
        else:
            # pandas/hats catalog
            mask = self.catalog[self.object_id_column] == object_id
            matching_indices = self.catalog.index[mask].tolist()
            if matching_indices:
                catalog_idx = matching_indices[0]

        if catalog_idx is None:
            raise ValueError(f"Object ID {object_id} not found in catalog")

        cutout_path = self._get_cutout_path(catalog_idx)
        if cutout_path.exists():
            # Load cached cutout
            cutout = torch.load(cutout_path, map_location="cpu", weights_only=True)

            # Apply band filtering if needed
            if self._is_filtering_bands and self._band_indices is not None:
                cutout = cutout[self._band_indices]

            return cutout
        else:
            # Cutout not downloaded yet, cannot load for cache
            raise FileNotFoundError(f"Cutout file {cutout_path} not found. Download cutouts first.")

    def __len__(self):
        """Return length of current catalog, not the full manifest."""
        return len(self.catalog)

    def _get_manifest_index_for_catalog_index(self, catalog_idx):
        """Map catalog index to manifest index when filtering is active."""
        if self._catalog_to_manifest_index_map is None:
            # No filtering - direct mapping
            return catalog_idx

        # Use pre-built mapping for efficiency
        if catalog_idx in self._catalog_to_manifest_index_map:
            return self._catalog_to_manifest_index_map[catalog_idx]

        raise ValueError(f"Catalog index {catalog_idx} not found in mapping")

    def get_image(self, idxs):
        """Fetch image cutout(s) for given index or indices, using caching and band filtering.

        Parameters:
        -----------
        idxs: int or slice or list
            Index or indices to fetch.

        Returns:
        --------
        torch.Tensor or list of torch.Tensor:
            Single cutout tensor or list of cutout tensors.
        """

        # Handle single index
        if isinstance(idxs, int):
            row = self.catalog[idxs] if isinstance(self.catalog, Table) else self.catalog.iloc[idxs]
            manifest_idx = self._get_manifest_index_for_catalog_index(idxs)
            return self._fetch_single_cutout(row, idx=idxs, manifest_idx=manifest_idx)

        # Handle multiple indices
        cutouts = []
        for idx in idxs:
            row = self.catalog[idx] if isinstance(self.catalog, Table) else self.catalog.iloc[idx]
            manifest_idx = self._get_manifest_index_for_catalog_index(idx)
            cutouts.append(self._fetch_single_cutout(row, idx=idx, manifest_idx=manifest_idx))

        return cutouts

    def __getitem__(self, idxs) -> dict:
        """Modified to pass index for saving cutouts.

        Parameters:
        -----------
        idxs: int or slice or list
            Index or indices to fetch.

        Returns:
        --------
        dict:
            Dictionary with key 'data' containing another dict of default data fields
            to return. Currently only 'image' is supported.
        """

        return {"data": {"image": self.get_image(idxs)}}

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
        for catalog_idx in indices:
            manifest_idx = self._get_manifest_index_for_catalog_index(catalog_idx)
            cutout_path = self._get_cutout_path(catalog_idx)

            # Check if file exists on disk
            if cutout_path.exists():
                continue

            # Check manifest status
            if self.catalog_type == "astropy":
                filename = self.manifest["filename"][manifest_idx]
            else:
                filename = self.manifest.iloc[manifest_idx]["filename"]

            # Skip if already attempted and failed (unless force_retry is True)
            if filename == "Attempted" and not force_retry:
                logger.debug(
                    f"Skipping previously failed download for catalog\
                            index {catalog_idx} (manifest index {manifest_idx})"
                )
                continue

            indices_to_download.append((catalog_idx, manifest_idx))

        if not indices_to_download:
            logger.info("All cutouts already downloaded")
            return

        # Determine number of workers
        if max_workers is None:
            max_workers = self._determine_numprocs_download()

        logger.info(f"Downloading {len(indices_to_download)} cutouts using {max_workers} threads.")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_single_cutout, catalog_idx, manifest_idx): (
                    catalog_idx,
                    manifest_idx,
                )
                for catalog_idx, manifest_idx in indices_to_download
            }

            with tqdm(total=len(indices_to_download), desc="Downloading cutouts") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        catalog_idx, manifest_idx = futures[future]
                        logger.error(
                            f"Failed to download cutout\
                                    catalog_idx={catalog_idx}, manifest_idx={manifest_idx}: {e}"
                        )
                        self._update_manifest_entry(manifest_idx, None, "Attempted", [])
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

    def _download_single_cutout(self, catalog_idx, manifest_idx):
        """Helper method to download a single cutout."""
        cutout_path = self._get_cutout_path(catalog_idx)
        if cutout_path.exists():
            return

        try:
            row = (
                self.catalog[catalog_idx]
                if isinstance(self.catalog, Table)
                else self.catalog.iloc[catalog_idx]
            )
            cutout, downloaded_bands = self._fetch_cutout_with_cache(row)

            #  Only save cutout and mark as successful if at least one band worked
            if len(downloaded_bands) == 0:
                # All bands failed - don't save file, mark as "Attempted"
                self._update_manifest_entry(manifest_idx, None, "Attempted", downloaded_bands)
                raise RuntimeError(
                    f"All bands failed for cutout\
                        catalog_idx={catalog_idx}, manifest_idx={manifest_idx}"
                )
            else:
                # At least some bands succeeded - save the cutout
                torch.save(cutout, cutout_path)
                filename = cutout_path.name
                self._update_manifest_entry(manifest_idx, cutout.shape, filename, downloaded_bands)

        except Exception as e:
            logger.error(
                f"Failed to download cutout\
                    catalog_idx={catalog_idx}, manifest_idx={manifest_idx}: {e}"
            )
            # Update manifest with failed attempt (all bands failed)
            self._update_manifest_entry(manifest_idx, None, "Attempted", [])
            raise

    def cache_info(self):
        """Get cache statistics."""
        return self._request_patch_cached.cache_info()

    def clear_cache(self):
        """Clear the LRU cache."""
        self._request_patch_cached.cache_clear()
        logger.info("Cleared patch cache")

    def manifest_stats(self):
        """Get manifest statistics including downloaded bands information."""
        with self._manifest_lock:
            if self.catalog_type == "astropy":
                successful = sum(
                    1 for filename in self.manifest["filename"] if filename and filename != "Attempted"
                )
                failed = sum(1 for filename in self.manifest["filename"] if filename == "Attempted")
                pending = sum(1 for filename in self.manifest["filename"] if not filename)
                expected_band_count = (
                    len(self._original_bands) if self._is_filtering_bands else len(self.BANDS)
                )

                # Add statistics about partial downloads (cutouts with missing bands)
                partial_downloads = sum(
                    1
                    for i, (filename, downloaded_bands) in enumerate(
                        zip(self.manifest["filename"], self.manifest["downloaded_bands"])
                    )
                    if filename
                    and filename != "Attempted"
                    and downloaded_bands
                    and len(downloaded_bands.split(",")) < expected_band_count
                )
            else:
                successful = sum(
                    1 for filename in self.manifest["filename"] if filename and filename != "Attempted"
                )
                failed = sum(1 for filename in self.manifest["filename"] if filename == "Attempted")
                pending = sum(1 for filename in self.manifest["filename"] if filename is None)
                expected_band_count = (
                    len(self._original_bands) if self._is_filtering_bands else len(self.BANDS)
                )

                partial_downloads = sum(
                    1
                    for i, (filename, downloaded_bands) in enumerate(
                        zip(self.manifest["filename"], self.manifest["downloaded_bands"])
                    )
                    if filename
                    and filename != "Attempted"
                    and downloaded_bands
                    and len(str(downloaded_bands).split(",")) < expected_band_count
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

    def band_filtering_info(self):
        """Get information about current band filtering configuration."""
        if not self._is_filtering_bands:
            return {
                "is_filtering": False,
                "requested_bands": list(self.BANDS),
                "original_bands": None,
                "filtered_bands": None,
                "band_indices": None,
            }

        return {
            "is_filtering": True,
            "requested_bands": list(self.BANDS),
            "original_bands": self._original_bands,
            "filtered_bands": self._filtered_bands,
            "band_indices": self._band_indices,
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

    def download_progress(self):
        """Get detailed download progress information."""
        stats = self.manifest_stats()

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

    def download_summary(self):
        """
        Get detailed download and band analysis, accounting for band filtering.
        """
        stats = self.manifest_stats()

        # Determine which bands to analyze based on filtering
        bands_to_analyze = self._filtered_bands if self._is_filtering_bands else list(self.BANDS)
        all_possible_bands = self._original_bands if self._is_filtering_bands else list(self.BANDS)

        # Analyze downloaded bands per cutout
        band_success_analysis = {band: 0 for band in bands_to_analyze}
        complete_downloads = 0

        if self.catalog_type == "astropy":
            downloaded_bands_entries = self.manifest["downloaded_bands"]
        else:
            downloaded_bands_entries = self.manifest["downloaded_bands"]

        for downloaded_bands_str in downloaded_bands_entries:
            if downloaded_bands_str and str(downloaded_bands_str).strip():
                downloaded_bands = [b.strip() for b in str(downloaded_bands_str).split(",") if b.strip()]

                # Filter to only bands we're interested in
                relevant_bands = [band for band in downloaded_bands if band in bands_to_analyze]

                # Count successful downloads per band
                for band in relevant_bands:
                    if band in band_success_analysis:
                        band_success_analysis[band] += 1

                # Count complete downloads (all requested bands present)
                if len(relevant_bands) == len(bands_to_analyze):
                    complete_downloads += 1

        filtering_info = "No filtering applied"
        if self._is_filtering_bands:
            filtering_info = f"Filtering {all_possible_bands} -> {bands_to_analyze}"

        return {
            "total_cutouts": stats["total"],
            "complete_downloads": complete_downloads,  # All requested bands present
            "partial_downloads": stats["partial_downloads"],  # Some requested bands missing
            "failed_downloads": stats["failed"],  # All bands failed
            "pending_downloads": stats["pending"],
            "band_success_counts": band_success_analysis,  # How many cutouts have each requested band
            "band_failure_counts": stats["band_failure_counts"],  # How many times each band failed
            "expected_bands": bands_to_analyze,
            "band_filtering_info": filtering_info,
            "percentage_complete": round(complete_downloads / stats["total"] * 100, 2)
            if stats["total"] > 0
            else 0,
        }
