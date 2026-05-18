"""Config migration: version 2 → version 3.

Moves the training shuffle option from ``[data_loader]`` to ``[train]``.
"""

from tomlkit.toml_document import TOMLDocument

from hyrax.config_migrations.migration_utils import migration_step, move_key


@migration_step(from_version=2, key_renames={"data_loader.shuffle": "train.shuffle"})
def move_data_loader_shuffle_to_train(cfg: TOMLDocument) -> TOMLDocument:
    """Move legacy ``[data_loader].shuffle`` to train-only ``[train].shuffle``."""
    move_key(cfg, "data_loader.shuffle", "train.shuffle")
    return cfg
