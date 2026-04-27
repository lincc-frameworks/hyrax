"""Migration engine, helpers, and registry for versioned config migrations."""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import tomlkit
from tomlkit.toml_document import TOMLDocument

from hyrax.config_utils import ConfigManager, parse_dotted_key

logger = logging.getLogger(__name__)

#: Mapping from source version to the migration step that upgrades it by one.
#: Populated at import time by the :func:`migration_step` decorator in each
#: versioned migration module.
MIGRATIONS: dict[int, "MigrationStep"] = {}


@dataclass(frozen=True)
class MigrationStep:
    """A single schema migration with optional key-rename metadata.

    Parameters
    ----------
    func : Callable[[TOMLDocument], TOMLDocument]
        The function that upgrades a config document by one version.
    key_renames : dict[str, str]
        Old dotted path -> new dotted path for every key renamed by this
        migration. Used by ``ConfigManager.set_config`` to warn callers
        who still reference the old names at runtime.
    """

    func: Callable[[TOMLDocument], TOMLDocument]
    key_renames: dict[str, str] = field(default_factory=dict)


def migration_step(from_version: int, key_renames: dict[str, str] | None = None):
    """Decorator that registers a migration function into :data:`MIGRATIONS`.

    Parameters
    ----------
    from_version : int
        The config schema version this function upgrades FROM. The target
        version is implicitly ``from_version + 1``.
    key_renames : dict[str, str], optional
        Old dotted path -> new dotted path for every key renamed by this
        migration.
    """

    def decorator(func: Callable[[TOMLDocument], TOMLDocument]):
        # Error if from_version is already registered
        if from_version in MIGRATIONS:
            existing_func = MIGRATIONS[from_version].func
            msg = (
                f"Duplicate migration from version: {from_version}! "
                f"Already registered: {existing_func.__module__}.{existing_func.__name__} "
                f"Attempted to register: {func.__module__}.{func.__name__}"
            )
            logger.error(msg)

        MIGRATIONS[from_version] = MigrationStep(func=func, key_renames=key_renames or {})
        return func

    return decorator


def rename_table(cfg: TOMLDocument | dict, old: str, new: str) -> bool:
    """Rename a top-level table from ``old`` to ``new``.

    If only ``old`` exists it is moved. If both exist, ``cfg[new]`` wins on
    leaf collisions; the two are deep-merged via `ConfigManager.merge_configs`
    and ``old`` is removed. If neither exists this is a no-op.

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

    msg = f"Migration: `[{old}]` has been renamed to `[{new}]`; "
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    logger.warning(msg)

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


def _build_deprecated_key_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for step in MIGRATIONS.values():
        result.update(step.key_renames)
    return result


def migrate_config(
    user_config: TOMLDocument,
    *,
    _migrations: dict[int, MigrationStep] | None = None,
    _target_version: int | None = None,
) -> TOMLDocument:
    """Upgrade a user config document to the current schema version.

    The document is mutated in place (and also returned for convenience). If
    ``config_version`` is absent it is assumed to be ``1`` — older configs
    predate the versioning field. If it is greater than the current schema
    version, a :class:`RuntimeError` is raised because the installed Hyrax
    does not know how to read the schema.

    Parameters
    ----------
    user_config : TOMLDocument
        The parsed user config. An empty document (the "no user config" case)
        is returned unchanged.
    _migrations : dict[int, MigrationStep], optional
        Override the global :data:`MIGRATIONS` registry (testing only).
    _target_version : int, optional
        Override the auto-derived target version (testing only).

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
    migrations = _migrations if _migrations is not None else MIGRATIONS
    current_config_version = (
        _target_version if _target_version is not None else (max(migrations.keys()) + 1 if migrations else 1)
    )

    # Empty user config (e.g. falling back to packaged defaults): nothing to do.
    if not user_config:
        return user_config

    # If user_config doesn't have config_version, assume it's the latest version
    user_version = user_config.pop("config_version", current_config_version)

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

    if user_version > current_config_version:
        raise RuntimeError(
            f"Config declares config_version = {user_version} but this Hyrax "
            f"install only understands up to config_version = "
            f"{current_config_version}. Upgrade with `pip install -U hyrax`."
        )

    current = user_version
    while current < current_config_version:
        step = migrations.get(current)
        if step is None:
            raise RuntimeError(
                f"No migration registered from config_version {current} to "
                f"{current + 1}. This is a Hyrax bug — please report it."
            )
        user_config = step.func(user_config)

        version_migration_complete_msg = (
            f"The configuration file has been migrated from version {current} to version {current + 1}. "
        )

        logger.warning(version_migration_complete_msg)

        current += 1

    final_migration_msg = (
        "All migrations complete. Your configuration file is now up to date with the latest schema. "
        "The runtime config saved in the output directory will reflect the new schema, "
        "and your original config file will remain unchanged on disk."
    )

    logger.warning(final_migration_msg)

    user_config["config_version"] = current_config_version
    return user_config
