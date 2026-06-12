"""Config migration: version 4 → version 5.

Removes the deprecated ``preload_cache`` and ``preload_threads`` keys from
``[data_set]``.  Cache preloading has been removed; use PyTorch DataLoader's
``num_workers`` and ``prefetch_factor`` instead.
"""

from tomlkit.toml_document import TOMLDocument

from hyrax.config_migrations.migration_utils import migration_step


@migration_step(from_version=4)
def remove_preload_config(cfg: TOMLDocument) -> TOMLDocument:
    """Remove deprecated preload_cache and preload_threads keys."""
    data_set = cfg.get("data_set")
    if isinstance(data_set, dict):
        data_set.pop("preload_cache", None)
        data_set.pop("preload_threads", None)
    return cfg
