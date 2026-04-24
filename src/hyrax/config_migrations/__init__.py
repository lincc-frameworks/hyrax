"""Versioned migrations for Hyrax user configuration files.

Hyrax tags its config schema with a top-level ``config_version`` scalar in
``hyrax_default_config.toml``. When a user loads an older config, the
migrations registered here run before the merge step in
:class:`hyrax.config_utils.ConfigManager`, bringing the user's document
forward one version at a time until it matches :data:`CURRENT_CONFIG_VERSION`.

Each migration step lives in its own descriptively-named module (e.g.
``v1_rename_model_inputs.py``) and self-registers via the
:func:`migration_step` decorator, which populates the :data:`MIGRATIONS` dict.
:data:`CURRENT_CONFIG_VERSION` is auto-derived from the highest registered
migration — developers do not bump it manually.

Adding a new migration:

1. Create ``src/hyrax/config_migrations/migrations/vN_description.py`` (e.g.
   ``v3_move_learning_rate.py``). Decorate the migration function with
   ``@migration_step(from_version=N, key_renames={...})``. Import the
   decorator and helpers from ``hyrax.config_migrations.migration_utils``.
   The module is auto-discovered — no import line needed elsewhere.
   ``CURRENT_CONFIG_VERSION`` and ``config_version`` in the default TOML are
   both stamped automatically at runtime.
2. Add a unit test in ``tests/hyrax/test_config_migrations.py``.
"""

# ruff: noqa: I001  — import order matters: machinery before migration modules

from hyrax.config_migrations.migration_utils import (  # noqa: F401
    MIGRATIONS,
    MigrationStep,
    _build_deprecated_key_map,
    migrate_config,
    migration_step,
    move_key,
    rename_table,
)

# Import migration modules to trigger @migration_step registration.
from hyrax.config_migrations.migrations import *  # noqa: F403

# Derived AFTER all migration modules are imported and registered.
CURRENT_CONFIG_VERSION: int = max(MIGRATIONS.keys()) + 1 if MIGRATIONS else 1
DEPRECATED_KEY_NAMES: dict[str, str] = _build_deprecated_key_map()

__all__ = [
    "CURRENT_CONFIG_VERSION",
    "DEPRECATED_KEY_NAMES",
    "MIGRATIONS",
    "MigrationStep",
    "migrate_config",
    "migration_step",
    "move_key",
    "rename_table",
]
