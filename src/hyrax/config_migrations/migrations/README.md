This directory contains all the migrations for the main hyrax configuration.

It is expected that each module in this directory map to a migration from version N
to version N+1.

The name of the modules isn't important aside from making it easier for a human to read.

The most important aspect is to correctly tag the migration function with the appropriate
`from_version`.

For instance to go from version N to N+1, the migration function might look like this:
```python
from tomlkit.toml_document import TOMLDocument
from hyrax.config_migrations.migration_utils import migration_step, move_key, rename_table

@migration_step(
    from_version=N, # <--- migrating from this version to N+1
    key_renames={
        "table_foo": "table_bar",
        "nested_1.sub_1": "nested_2.sub_1"
    }
)
def _migrate_vN_to_vN1(cfg: TOMLDocument) -> TOMLDocument:
    """Rename the legacy ``[table_foo]`` table to ``[table_bar]``,
    and move `sub_1` from `nested_1` to `nested_2`.
    """

    rename_table(cfg, "table_foo", "table_bar")
    move_key(cfg, "nested_1.sub_1", "nested_2.sub_1")

    return cfg
```
