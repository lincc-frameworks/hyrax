"""Config migration: version 4 → version 5.

Migrates the deprecated ``preload_cache`` and ``preload_threads`` keys from
``[data_set]`` into ``[data_loader].num_workers``.  Cache preloading has been
replaced by PyTorch DataLoader's built-in ``num_workers`` / ``prefetch_factor``.

If the user explicitly set ``preload_threads`` to a value other than the old
default of 50, that value is carried forward as ``num_workers``.  Otherwise both
keys are simply removed.
"""

import tomlkit
from tomlkit.toml_document import TOMLDocument

from hyrax.config_migrations.migration_utils import migration_step

# The old default for preload_threads was 50, tuned for UW's HYAK Klone HPC
# filesystem where I/O is extremely slow and lightweight threads were cheap.
# num_workers spawns full subprocesses, so 50 would be wildly inappropriate
# for most systems. Treat 50 as "user never customized this."
_OLD_DEFAULT_PRELOAD_THREADS = 50


@migration_step(from_version=4)
def remove_preload_config(cfg: TOMLDocument) -> TOMLDocument:
    """Migrate preload config to ``[data_loader].num_workers``."""
    data_set = cfg.get("data_set")
    if not isinstance(data_set, dict):
        return cfg

    preload_cache = data_set.pop("preload_cache", None)
    preload_threads = data_set.pop("preload_threads", None)

    if not preload_cache or preload_threads is None or preload_threads == _OLD_DEFAULT_PRELOAD_THREADS:
        return cfg

    data_loader = cfg.get("data_loader")
    if data_loader is None:
        data_loader = tomlkit.table()
        cfg["data_loader"] = data_loader

    existing = data_loader.get("num_workers", 0)
    data_loader["num_workers"] = max(existing + preload_threads, 1)

    return cfg
