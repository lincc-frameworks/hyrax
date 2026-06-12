# Augmentation Cache Restructuring Plan

## Summary

Restructure DataCache for per-dataset caching to support the `row_cache_key` dataset
interface. Remove the preload thread (replaced by PyTorch DataLoader `num_workers`).
Move cache-key dispatch logic into DataCache so DataProvider gets simpler, not more complex.

## Context

V1 augmentation (implemented in PR #942) caches `get_<field>` results in a single
flat map keyed by DataProvider index. Augmentation runs post-cache and is never
cached. This works when all datasets share the same index space, but breaks when:

- Multiple datasets have different index mappings (joins).
- A dataset wants to cache augmented results via `row_cache_key`.
- The preload thread assumes sequential 0..len access but the sampler
  (WeightedRandomSampler on `awo/splits-and-balancing-spec`) picks random indices.

This plan implements the `row_cache_key` interface from `specs/augmentation.md`,
restructures DataCache to use per-dataset cache maps, and removes the preload thread.

---

## Step 1: Add `row_cache_key` to HyraxDataset

**File:** `src/hyrax/datasets/dataset_registry.py`

Add one method to `HyraxDataset`:

```python
def row_cache_key(self, idx: int, rng_seed: np.int64 | None = None) -> np.int64 | None:
    """Return a cache key for this row, or None to skip caching.

    Parameters
    ----------
    idx : int
        The dataset-local index.
    rng_seed : np.int64 | None
        When augmentation is active, the rng_seed that would be passed to
        ``augment_<field>``.  ``None`` for non-augmented access.

    Returns
    -------
    np.int64 | None
        Cache key, or ``None`` to skip caching for this call.
    """
    return np.int64(idx) if rng_seed is None else None
```

Add `import numpy as np` at the top of the file (it already imports `numpy.typing`,
so add the numpy import next to it).

**Tests:** `tests/hyrax/test_augmentation.py` (add to existing file)
- Default `row_cache_key(5)` returns `np.int64(5)`.
- Default `row_cache_key(5, rng_seed=np.int64(42))` returns `None`.
- Subclass override is respected.

---

## Step 2: Remove preload thread references outside DataCache

**Important:** Steps 2 and 3 must be treated as a single atomic change to
`data_cache.py`. There is no valid intermediate state where preload is removed
but the constructor still takes a `DataProvider`. Step 2 covers changes
**outside** `data_cache.py`; Step 3 covers the full rewrite of `data_cache.py`
(which includes removing all preload code).

### 2a: Update DataProvider

**File:** `src/hyrax/datasets/data_provider.py`

Remove the `self.data_cache.start_preload_thread()` call in `__init__` (currently
on the line after `self.data_cache = DataCache(config, self)`). The DataCache
constructor call itself changes in Step 4.

### 2b: Update config and migration

**File:** `src/hyrax/hyrax_default_config.toml`

Remove these lines (and their preceding comments) from the `[data_set]` section:
```toml
# If `true`, preload the in memory cache ...
preload_cache = false

# Number of threads to use for cache preload ...
preload_threads = 50
```

**File:** `src/hyrax/config_migrations/migrations/003_remove_preload_config.py` (new)

```python
"""Config migration: version 3 → version 4.

Removes the deprecated ``preload_cache`` and ``preload_threads`` keys from
``[data_set]``.  Cache preloading has been removed; use PyTorch DataLoader's
``num_workers`` and ``prefetch_factor`` instead.
"""

from tomlkit.toml_document import TOMLDocument

from hyrax.config_migrations.migration_utils import migration_step


@migration_step(from_version=3)
def remove_preload_config(cfg: TOMLDocument) -> TOMLDocument:
    """Remove deprecated preload_cache and preload_threads keys."""
    data_set = cfg.get("data_set")
    if isinstance(data_set, dict):
        data_set.pop("preload_cache", None)
        data_set.pop("preload_threads", None)
    return cfg
```

`key_renames` defaults to `None` in `migration_step`, so omitting it is fine.

### 2c: Update references across codebase

**File:** `src/hyrax/datasets/inference_dataset.py` (line 94)

Remove:
```python
self._original_dataset_config["data_set"]["preload_cache"] = False
```

**File:** `src/hyrax/datasets/fits_image_dataset.py` (line 52 area)

Update the docstring to remove the reference to `preload_cache`. Change:
```
If your dataset does not fit in memory on your system, we recommend setting
``h.config["data_set"]["use_cache"]`` and ``h.config["data_set"]["preload_cache"]`` to ``False``.
Both are ``True`` by default. The former caches ...
```
To:
```
If your dataset does not fit in memory on your system, we recommend setting
``h.config["data_set"]["use_cache"]`` to ``False``.
This caches all tensors read during an epoch into system RAM, with the
intent of speeding up later epochs of training if your disk has low bandwidth.
This will result in crashes if there is not enough room in your system RAM
for the entire dataset.
```

**File:** `tests/hyrax/test_data_provider.py` (lines 107, 122)

Remove these lines:
```python
h.config["data_set"]["preload_cache"] = False  # This reduces warnings on this test
```

**File:** `tests/hyrax/test_lsst_butler_mocks.py` (line 78)

Remove `"preload_cache": True,` from the config dict.

**File:** `tests/hyrax/mocks/lsst_butler_fixtures.py` (line 75)

Remove `"preload_cache": True,` from the config dict.

**File:** `benchmarks/data_cache_benchmarks.py`

This entire benchmark file tests preload performance. Rewrite it to benchmark
per-dataset cache performance instead:
- Remove `time_preload_cache_hsc1k` and `track_cache_hsc1k_hyrax_size_undercount`.
- Remove `setup` method that sets `preload_cache = True`.
- Keep `setup_cache` but remove the `preload_cache = False` line.
- Add a new benchmark `time_cache_fill_hsc1k` that iterates over the dataset
  calling `data_provider[i]` for all indices, measuring the time to fill
  the cache through normal access.
- Add a new benchmark `time_cache_hit_hsc1k` that calls `data_provider[0]`
  after the cache is filled, measuring cache-hit performance.

---

## Step 3: Rewrite DataCache (atomic — includes preload removal)

**File:** `src/hyrax/datasets/data_cache.py`

This step is a complete rewrite of DataCache. All preload code, the DataProvider
backpointer, and the single-map architecture are removed in one pass.

### 3a: Remove imports that are no longer needed

Remove these imports entirely:
- `from collections.abc import Iterable` (only used in `_lazy_map_executor`)
- `from concurrent.futures import Executor` (only used in `_lazy_map_executor`)
- `from threading import Thread` (only used by preload thread)

Also remove:
- `from hyrax.datasets.data_provider import DataProvider` (old constructor type hint)

Keep: `logging`, `time`, `from numbers import Number`, `from sys import getsizeof`,
`from typing import Any`, `import numpy as np`,
`from hyrax.tensorboardx_logger import get_tensorboard_logger`.

### 3b: New constructor

Replace the entire `__init__` method — new signature, new body. The old constructor
accepted `(config, data_provider: DataProvider)`. The new one accepts per-dataset
information directly. All of the following old instance variables are gone:
`_max_length`, `_resolve_data_func`, `_data_provider`, `_preload_cache`,
`_preload_thread`, `_preload_threads`.

```python
def __init__(
    self,
    config: dict,
    datasets: dict[str, "HyraxDataset"],
    augment_active: dict[str, bool],
):
    """Initialize the DataCache.

    Parameters
    ----------
    config : dict
        The Hyrax configuration.
    datasets : dict[str, HyraxDataset]
        Mapping of friendly_name to dataset instance. Used to call
        ``row_cache_key`` during cache operations.
    augment_active : dict[str, bool]
        Mapping of friendly_name to whether augmentation is active
        for that dataset. When True, ``try_fetch`` will attempt a
        two-level lookup (augmented key first, then base key).
    """
    self._use_cache = config["data_set"]["use_cache"]
    self._datasets = datasets
    self._augment_active = augment_active

    self._data_size_bytes = 0
    self._insert_count = 0
    self.logging_interval = 1000

    # Per-dataset cache maps: friendly_name -> {cache_key: field_data_dict}
    self._cache_maps: dict[str, dict[np.int64, dict]] = {
        name: {} for name in datasets
    }
```

The `"HyraxDataset"` string annotation avoids a circular import. Do NOT add
`from hyrax.datasets.dataset_registry import HyraxDataset` at module level.

### 3c: Remove old methods

Delete entirely:
- `_idx_check` — cache keys from `row_cache_key` are arbitrary `np.int64` values,
  so bounds checking is not meaningful.
- `start_preload_thread`
- `_preload_tensor_cache`
- `_lazy_map_executor`

### 3d: New `try_fetch` method

Replace `try_fetch(self, idx)` with:

```python
def try_fetch(
    self,
    friendly_name: str,
    real_idx: int,
    rng_seed: np.int64 | None = None,
) -> tuple[dict | None, bool]:
    """Try to fetch cached data for a single dataset.

    When augmentation is active for this dataset and ``rng_seed`` is not
    None, this method first tries the augmented cache key. On miss, it
    falls back to the base cache key. When augmentation is not active,
    only the base key is tried.

    Parameters
    ----------
    friendly_name : str
        The dataset friendly name.
    real_idx : int
        The dataset-local index.
    rng_seed : np.int64 | None
        The augmentation RNG seed, or None for non-augmented access.

    Returns
    -------
    tuple[dict | None, bool]
        ``(data, already_augmented)`` where ``data`` is the cached
        field dict or ``None`` on miss, and ``already_augmented``
        indicates whether the cached data includes augmentation.
    """
    if not self._use_cache:
        return None, False

    dataset = self._datasets[friendly_name]
    cache_map = self._cache_maps[friendly_name]

    # When augmentation is active, try augmented key first
    if self._augment_active.get(friendly_name, False) and rng_seed is not None:
        aug_key = dataset.row_cache_key(real_idx, rng_seed)
        if aug_key is not None:
            cached = cache_map.get(aug_key)
            if cached is not None:
                return cached, True

    # Try base key
    base_key = dataset.row_cache_key(real_idx)
    if base_key is not None:
        cached = cache_map.get(base_key)
        if cached is not None:
            return cached, False

    return None, False
```

### 3e: New `insert` method

Replace `insert_into_cache(self, idx, data)` with:

```python
def insert(
    self,
    friendly_name: str,
    real_idx: int,
    rng_seed: np.int64 | None,
    data: dict[str, Any],
):
    """Insert field data into the cache for a single dataset.

    Calls ``row_cache_key`` to determine the cache key. If the key is
    ``None``, this is a no-op (the dataset opted out of caching for
    this call).

    Parameters
    ----------
    friendly_name : str
        The dataset friendly name.
    real_idx : int
        The dataset-local index.
    rng_seed : np.int64 | None
        The augmentation RNG seed, or None for base data.
    data : dict[str, Any]
        The field data dict to cache.
    """
    start_time = time.monotonic_ns()
    prefix = self.__class__.__name__

    if not self._use_cache:
        return

    dataset = self._datasets[friendly_name]
    cache_key = dataset.row_cache_key(real_idx, rng_seed)
    if cache_key is None:
        return

    cache_map = self._cache_maps[friendly_name]
    self._insert_count += 1
    old_value = cache_map.get(cache_key)
    if old_value is not None:
        self._data_size_bytes -= DataCache._data_size(old_value)

    cache_map[cache_key] = data
    self._data_size_bytes += DataCache._data_size(data)
    tensorboardx_logger.log_duration_ts(f"{prefix}/cache_insert_s", start_time)
    if self._insert_count % self.logging_interval == 0 and self._insert_count != 0:
        tensorboardx_logger.log_scalar_ts(f"{prefix}/cache_count", self._insert_count)
        tensorboardx_logger.log_scalar_ts(f"{prefix}/cache_bytes", self._data_size_bytes)
```

### 3f: Keep `_data_size` unchanged

The static `_data_size` method is unchanged — it works on any dict structure.

### 3g: Update the class docstring

Rewrite the class-level docstring to reflect per-dataset caching and the removal
of preload. Remove all references to preloading threads. Mention that caching is
now per-dataset and uses `row_cache_key` for key computation.

---

## Step 4: Update DataProvider to use new DataCache API

**File:** `src/hyrax/datasets/data_provider.py`

### 4a: Update DataCache construction in `__init__`

Note on `augment_enabled` vs `augment_active`: In DataProvider,
`self.augment_enabled[friendly_name]` is a `dict[str, bool]` (per-field config
from V2). Its truthiness (non-empty dict = True) is used in `resolve_data` to
check whether a dataset has any augmentation. The `augment_active` dict passed
to DataCache is a simple `bool` per dataset, derived from whether any augment
getters were wired up. These are different types serving different roles.

Replace:
```python
from hyrax.datasets.data_cache import DataCache
self.data_cache = DataCache(config, self)
self.data_cache.start_preload_thread()
```

With:
```python
from hyrax.datasets.data_cache import DataCache
augment_active = {
    fn: bool(self.augment_getters.get(fn))
    for fn in self.prepped_datasets
}
self.data_cache = DataCache(config, self.prepped_datasets, augment_active)
```

### 4b: Extract `_apply_augmentation` helper

Add a new private method to DataProvider to apply augmentation to a single
dataset's field data. This extracts logic currently inline in `resolve_data`:

```python
def _apply_augmentation(
    self,
    friendly_name: str,
    base_data: dict[str, Any],
    real_idx: int,
    rng_seed: np.int64,
) -> dict[str, Any]:
    """Apply augmentation to base field data for a single dataset.

    Passes read-only ndarray views to augment functions to protect
    cached base data from mutation.
    """
    new_fields: dict[str, Any] = {}
    for field, value in base_data.items():
        augment_fn = self.augment_getters.get(friendly_name, {}).get(field)
        if augment_fn is not None and isinstance(value, np.ndarray):
            value = value.view()
            value.flags.writeable = False
        new_fields[field] = (
            augment_fn(value, real_idx, rng_seed) if augment_fn is not None else value
        )
    return new_fields
```

### 4c: Rewrite `resolve_data`

**Keep the existing docstring** — it describes the return format and join behavior,
which are still accurate. Only replace the function body.

Replace the current `resolve_data` implementation with a per-dataset loop
that delegates cache decisions to DataCache. The new flow:

```python
def resolve_data(self, idx: int) -> dict[str, dict[str, Any] | str | None]:
    start_time = time.monotonic_ns()
    prefix = self.__class__.__name__

    rng_seed = self._augment_rng_seed() if self._has_any_augmentation else None

    # Pre-fetch primary object ID when any joins are configured.
    if self._join_maps:
        primary_id_getter = self.dataset_getters[self.primary_dataset][
            self.primary_dataset_id_field_name
        ]
        object_id_str = str(primary_id_getter(idx))
    else:
        object_id_str = None

    result: dict[str, dict[str, Any] | str | None] = {}
    had_any_miss = False

    for friendly_name, fields in self.requested_fields.items():
        getters = self.dataset_getters[friendly_name]

        # Determine real index (join mapping).
        if friendly_name in self._join_maps:
            real_idx = self._join_maps[friendly_name].get(object_id_str)
            if real_idx is None:
                result[friendly_name] = None
                continue
        else:
            real_idx = idx

        # Determine effective rng_seed for this dataset.
        # augment_enabled[fn] is a dict[str, bool] (per-field V2 config);
        # its truthiness (non-empty dict) indicates any augmentation is active.
        effective_rng = rng_seed if self.augment_enabled.get(friendly_name) else None

        # Ask DataCache — it handles row_cache_key dispatch and the
        # two-level lookup (augmented then base) internally.
        cached_data, already_augmented = self.data_cache.try_fetch(
            friendly_name, real_idx, effective_rng
        )

        if cached_data is not None and (already_augmented or effective_rng is None):
            # Final data — either augmented data from cache, or base data
            # when no augmentation is needed.
            result[friendly_name] = cached_data
        elif cached_data is not None:
            # Base data from cache, augmentation still needed.
            augmented = self._apply_augmentation(
                friendly_name, cached_data, real_idx, rng_seed
            )
            self.data_cache.insert(friendly_name, real_idx, rng_seed, augmented)
            result[friendly_name] = augmented
        else:
            # Full cache miss — call get_<field> methods.
            had_any_miss = True
            base_data = {field: getters[field](real_idx) for field in fields}
            self.data_cache.insert(friendly_name, real_idx, None, base_data)

            if effective_rng is not None:
                augmented = self._apply_augmentation(
                    friendly_name, base_data, real_idx, rng_seed
                )
                self.data_cache.insert(friendly_name, real_idx, rng_seed, augmented)
                result[friendly_name] = augmented
            else:
                result[friendly_name] = base_data

    # Add object_id (same logic as current code).
    if self.primary_dataset:
        if self.primary_dataset_id_field_name not in result.get(self.primary_dataset, {}):
            if object_id_str is not None:
                object_id = object_id_str
            else:
                primary_getter = self.dataset_getters[self.primary_dataset]
                object_id = str(primary_getter[self.primary_dataset_id_field_name](idx))
        else:
            object_id = result[self.primary_dataset][self.primary_dataset_id_field_name]
        result["object_id"] = str(object_id)

    # Timing metrics.
    if had_any_miss:
        tensorboardx_logger.log_duration_ts(f"{prefix}/cache_miss_s", start_time)
    else:
        tensorboardx_logger.log_duration_ts(f"{prefix}/cache_hit_s", start_time)

    return result
```

Key differences from current code:
- Cache interaction is per-dataset, not whole-row.
- The augmentation pass is folded into the per-dataset loop (no separate sweep).
- `_apply_augmentation` helper keeps the loop body readable.
- `augmentation_s` timing is removed as a separate metric since augmentation is
  now interleaved with cache lookups. If desired, time the `_apply_augmentation`
  calls and log them, but this is optional.
- The `had_any_miss` flag drives the top-level timing metric. A resolve_data call
  where every dataset hit cache logs `cache_hit_s`; any miss logs `cache_miss_s`.

---

## Step 5: Tests

**File:** `tests/hyrax/test_augmentation.py` (add to existing file)

### 5a: Test dataset with custom `row_cache_key`

Create a test dataset that caches augmented results:

```python
class CachingAugmentDataset(HyraxRandomDataset):
    """Dataset that caches augmented results by including rng_seed in cache key."""

    def augment_image(self, data, idx, rng_seed):
        return -data

    def row_cache_key(self, idx, rng_seed=None):
        if rng_seed is None:
            return np.int64(idx)
        # Cache augmented data under a different key derived from idx + rng_seed.
        return np.int64(idx * 1_000_000 + (rng_seed % 1_000_000))
```

### 5b: Test cases

1. **Per-dataset cache hit/miss:** Two datasets in one DataProvider. Access an
   index, verify both datasets' data is cached independently. Invalidate one
   dataset's cache (by using a different `row_cache_key` subclass), verify the
   other dataset still has a cache hit.

2. **Augmented data not cached by default:** With default `row_cache_key`,
   verify that calling `resolve_data` twice with augmentation enabled produces
   different augmented results (different rng_seed each call) but the base
   data is cached (get methods called only once). Use a spy/counter on the
   `get_image` method to verify call count.

3. **Augmented data cached with custom `row_cache_key`:** Test DataCache
   directly (not through DataProvider) so you can control the rng_seed.
   Construct a DataCache with a `CachingAugmentDataset` instance and
   `augment_active=True`. Call `insert(fn, idx=0, rng_seed=np.int64(42), data)`
   then `try_fetch(fn, idx=0, rng_seed=np.int64(42))` and verify
   `(data, True)` is returned. Also verify that a different rng_seed
   (e.g. `np.int64(99)`) misses the augmented cache but still hits the
   base cache if base data was also inserted.

4. **`row_cache_key` returning None skips cache:** Create a dataset whose
   `row_cache_key` always returns `None`. Verify data is never cached
   (get methods called every time).

5. **Mixed datasets:** One dataset with augmentation, one without. Verify
   the non-augmented dataset caches normally (key = idx), and the augmented
   dataset caches base data but not augmented data (default behavior).

6. **Join-map index translation:** With a joined secondary dataset, verify
   that `row_cache_key` receives the dataset-local index (from the join map),
   not the DataProvider-level index.

### 5c: Update existing tests

Existing tests in `test_data_provider.py` and `test_augmentation.py` that construct
DataCache or DataProvider may need updating for the new constructor signature.
Search for `DataCache(` across the test suite and update each call site.

---

## Step 6: Run checks

After all changes:
```bash
ruff check src/ tests/ && ruff format src/ tests/
pre-commit run --all-files
python -m pytest -m "not slow"
```

---

## Implementation Order

1. **Step 1** — Add `row_cache_key` to HyraxDataset. No dependencies.
2. **Step 2** — Remove preload references outside DataCache (config, tests, docs).
   No dependency on Step 1.
3. **Steps 3 + 4 together** — Rewrite DataCache and update DataProvider. These
   are tightly coupled: changing DataCache's API requires updating its sole
   caller simultaneously. Do them in a single pass. Depends on Step 1.
4. **Step 5** — Tests. Depends on Steps 3+4.
5. **Step 6** — Lint and test.

Steps 1 and 2 have no dependency on each other and can be done in either order.

---

## Design Decisions

1. **Preload thread removed, not moved.** PyTorch DataLoader's `num_workers` and
   `prefetch_factor` already solve I/O prefetching and know the actual access
   pattern from the sampler. The preload thread's sequential 0..len strategy
   is incompatible with WeightedRandomSampler.

2. **DataCache owns `row_cache_key` dispatch.** DataProvider calls
   `data_cache.try_fetch(friendly_name, real_idx, rng_seed)` and DataCache
   internally calls the dataset's `row_cache_key`. This keeps cache-key logic
   out of DataProvider.

3. **Two-level lookup only when needed.** DataCache knows at construction time
   which datasets have augmentation enabled. For datasets without augmentation,
   `try_fetch` does a single lookup (base key only). The augmented-key lookup
   is only attempted when `augment_active[friendly_name]` is True and
   `rng_seed` is not None.

4. **Per-dataset cache maps.** Each dataset gets its own `dict[np.int64, dict]`.
   This avoids key collisions between datasets and makes size tracking
   straightforward. The aggregate `_data_size_bytes` covers all maps.

5. **`_idx_check` removed.** Cache keys from `row_cache_key` are arbitrary
   `np.int64` values (could include rng_seed hash), so bounds checking against
   dataset length is not meaningful.

6. **Augmentation folded into per-dataset loop.** Instead of a separate
   augmentation sweep over the assembled data dict, augmentation is applied
   per-dataset inside the cache loop. This eliminates the second pass and
   the `augmentation_s` metric (which can be re-added per-dataset if needed).

7. **Config migration for preload keys.** A migration (003) removes
   `preload_cache` and `preload_threads` from user TOML files. Code stops
   reading them. The default config entries are deleted.
