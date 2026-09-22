"""Config migration: version 3 → version 4.

Moves per-dataset ``split_fraction`` from ``[data_request]`` into the new
top-level ``[split]`` table.
"""

import tomlkit
from tomlkit.toml_document import TOMLDocument

from hyrax.config_migrations.migration_utils import migration_step


@migration_step(from_version=3)
def move_split_fraction_to_split(cfg: TOMLDocument) -> TOMLDocument:
    """Move per-dataset ``split_fraction`` from [data_request] into [split]."""
    data_request = cfg.get("data_request")
    if not data_request:
        return cfg

    split_tbl = cfg.get("split", tomlkit.table())
    for group_name, group in data_request.items():
        if not isinstance(group, dict):
            continue
        for _friendly, dsdef in group.items():
            if isinstance(dsdef, dict) and "split_fraction" in dsdef:
                # The Pydantic schema guarantees split_fraction only sits on the
                # group's primary dataset, so the last/only write per group wins.
                split_tbl[group_name] = dsdef.pop("split_fraction")
    if len(split_tbl):
        cfg["split"] = split_tbl
    return cfg
