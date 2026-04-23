"""Config migration: version 1 → version 2.

Renames the legacy ``[model_inputs]`` table to ``[data_request]``.
"""

from tomlkit.toml_document import TOMLDocument

from hyrax.config_migrations._machinery import migration_step, rename_table


@migration_step(from_version=1, key_renames={"model_inputs": "data_request"})
def _migrate_v1_to_v2(cfg: TOMLDocument) -> TOMLDocument:
    """Rename the legacy ``[model_inputs]`` table to ``[data_request]``."""
    rename_table(cfg, "model_inputs", "data_request")
    return cfg
