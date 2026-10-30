"""Config migration: version 6 → version 7.

Renames the ``[reduce]`` table with ``[reduce_dimensions]``.
This is to keep the naming consistent with the config and verb.
"""

from tomlkit.toml_document import TOMLDocument

from hyrax.config_migrations.migration_utils import migration_step, rename_table


@migration_step(from_version=6, key_renames={"reduce": "reduce_dimensions"})
def rename_reduce_to_reduce_dimensions(cfg: TOMLDocument) -> TOMLDocument:
    """Renames the ``[reduce]`` table with ``[reduce_dimensions]``."""
    rename_table(cfg, "reduce", "reduce_dimensions")
    return cfg
