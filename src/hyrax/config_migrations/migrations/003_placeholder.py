"""Config migration: version 3 → version 4.

No-op placeholder.  The real migration 003 lives on the splits-and-balancing
branch.  Remove this file during merge conflict resolution when that branch
lands.
"""

from tomlkit.toml_document import TOMLDocument

from hyrax.config_migrations.migration_utils import migration_step


@migration_step(from_version=3)
def placeholder(cfg: TOMLDocument) -> TOMLDocument:
    """No-op placeholder; remove during merge with splits-and-balancing branch."""
    return cfg
