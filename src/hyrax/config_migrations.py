"""Versioned migrations for Hyrax user configuration files.

Hyrax tags its config schema with a top-level ``config_version`` scalar in
``hyrax_default_config.toml``. When a user loads an older config, the
migrations registered here run before the merge step in
:class:`hyrax.config_utils.ConfigManager`, bringing the user's document
forward one version at a time until it matches :data:`CURRENT_CONFIG_VERSION`.

Developers renaming a table or key should:

1. Add a new migration function ``_migrate_vN_to_vN_plus_1``.
2. Register it in :data:`MIGRATIONS` under its source version.
3. Bump :data:`CURRENT_CONFIG_VERSION` and update the ``config_version`` value
   in ``hyrax_default_config.toml``.
4. Add a unit test in ``tests/hyrax/test_config_migrations.py``.
"""

import logging
import warnings
from collections.abc import Callable

import tomlkit
from tomlkit.toml_document import TOMLDocument

from hyrax.config_utils import ConfigManager, parse_dotted_key

__all__ = [
    "CURRENT_CONFIG_VERSION",
    "MIGRATIONS",
    "migrate_config",
    "rename_table",
    "move_key",
]

#: The highest config schema version understood by this Hyrax install.
CURRENT_CONFIG_VERSION: int = 2

logger = logging.getLogger(__name__)


def rename_table(cfg: TOMLDocument | dict, old: str, new: str) -> bool:
    """Rename a top-level table from ``old`` to ``new``.

    If only ``old`` exists it is moved. If both exist, ``cfg[new]`` wins on
    leaf collisions; the two are deep-merged via
    :meth:`ConfigManager.merge_configs` and ``old`` is removed. If neither
    exists this is a no-op.

    Parameters
    ----------
    cfg : TOMLDocument or dict
        The user config document to mutate in place.
    old : str
        The table name that is being retired.
    new : str
        The new table name to adopt.

    Returns
    -------
    bool
        ``True`` iff ``old`` was present and something was renamed (so the
        caller can emit a deprecation warning conditionally).
    """
    if old not in cfg:
        return False

    old_value = cfg[old]
    if new in cfg:
        existing = cfg[new]
        if isinstance(existing, dict) and isinstance(old_value, dict):
            cfg[new] = ConfigManager.merge_configs(old_value, existing)
        # If either side is a scalar, the existing ``new`` key wins unchanged.
    else:
        cfg[new] = old_value

    del cfg[old]
    return True


def move_key(cfg: TOMLDocument | dict, old_path: str, new_path: str) -> bool:
    """Move a nested key from ``old_path`` to ``new_path``.

    Paths are dotted strings parsed by
    :func:`hyrax.config_utils.parse_dotted_key`, so quoted segments like
    ``"'torch.optim.Adam'.lr"`` are supported.

    Parameters
    ----------
    cfg : TOMLDocument or dict
        The user config document to mutate in place.
    old_path : str
        The dotted path to the existing key.
    new_path : str
        The dotted path to write the value to. Intermediate tables are
        created as needed.

    Returns
    -------
    bool
        ``True`` iff ``old_path`` was present and the value was moved.
    """
    old_parts = parse_dotted_key(old_path)
    new_parts = parse_dotted_key(new_path)
    if not old_parts or not new_parts:
        return False

    # Walk to the parent of old_parts without autovivifying.
    parent: dict = cfg
    for key in old_parts[:-1]:
        if not isinstance(parent, dict) or key not in parent:
            return False
        parent = parent[key]
    leaf = old_parts[-1]
    if not isinstance(parent, dict) or leaf not in parent:
        return False
    value = parent[leaf]
    del parent[leaf]

    # Autovivify the new parent chain.
    new_parent: dict = cfg
    for key in new_parts[:-1]:
        if key not in new_parent or not isinstance(new_parent[key], dict):
            new_parent[key] = tomlkit.table() if isinstance(cfg, TOMLDocument) else {}
        new_parent = new_parent[key]
    new_parent[new_parts[-1]] = value
    return True


def _migrate_v1_to_v2(cfg: TOMLDocument) -> TOMLDocument:
    """Rename the legacy ``[model_inputs]`` table to ``[data_request]``."""
    if rename_table(cfg, "model_inputs", "data_request"):
        msg = (
            "[model_inputs] has been renamed to [data_request]; update your "
            "config file. Hyrax has migrated the value for this run."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
        logger.warning(msg)
    return cfg


#: Mapping from source version to the function that upgrades it by one step.
#: A missing entry for version ``N`` means there is no ``N → N+1`` migration
#: and :func:`migrate_config` will raise ``RuntimeError``.
MIGRATIONS: dict[int, Callable[[TOMLDocument], TOMLDocument]] = {
    1: _migrate_v1_to_v2,
}


def migrate_config(user_config: TOMLDocument) -> TOMLDocument:
    """Upgrade a user config document to :data:`CURRENT_CONFIG_VERSION`.

    The document is mutated in place (and also returned for convenience). If
    ``config_version`` is absent it is assumed to be ``1`` — older configs
    predate the versioning field. If it is greater than
    :data:`CURRENT_CONFIG_VERSION`, a :class:`RuntimeError` is raised because
    the installed Hyrax does not know how to read the schema.

    Parameters
    ----------
    user_config : TOMLDocument
        The parsed user config. An empty document (the "no user config" case)
        is returned unchanged.

    Returns
    -------
    TOMLDocument
        The upgraded document with ``config_version`` stamped to the current
        value.

    Raises
    ------
    RuntimeError
        If ``user_config`` declares a version newer than this Hyrax can
        understand, or if a migration step is missing from :data:`MIGRATIONS`.
    """
    # Empty user config (e.g. falling back to packaged defaults): nothing to do.
    if not user_config:
        return user_config

    user_version = user_config.pop("config_version", 1)
    # Reject booleans (int subclass in Python — True/False would otherwise
    # sneak through as 1/0) and any non-int type (floats, strings, ...).
    # tomlkit parses TOML integers into a subclass of int, so tomlkit-parsed
    # ``config_version = 2`` still satisfies isinstance(..., int).
    if isinstance(user_version, bool) or not isinstance(user_version, int):
        raise RuntimeError(
            f"config_version must be a non-boolean integer, got {user_version!r} "
            f"(type {type(user_version).__name__}). Set it to a supported "
            "integer schema version (>= 1)."
        )

    if user_version < 1:
        raise RuntimeError(
            f"config_version must be >= 1, got {user_version}. Version 1 is the lowest supported schema."
        )

    if user_version > CURRENT_CONFIG_VERSION:
        raise RuntimeError(
            f"Config declares config_version = {user_version} but this Hyrax "
            f"install only understands up to config_version = "
            f"{CURRENT_CONFIG_VERSION}. Upgrade with `pip install -U hyrax`."
        )

    current = user_version
    while current < CURRENT_CONFIG_VERSION:
        migration = MIGRATIONS.get(current)
        if migration is None:
            raise RuntimeError(
                f"No migration registered from config_version {current} to "
                f"{current + 1}. This is a Hyrax bug — please report it."
            )
        user_config = migration(user_config)
        current += 1

    user_config["config_version"] = CURRENT_CONFIG_VERSION
    return user_config
