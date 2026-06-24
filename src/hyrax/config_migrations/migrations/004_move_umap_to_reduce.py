"""Config migration: version 4 → version 5.

Move the legacy ``[umap]`` and ``[umap.UMAP]`` to be under ``[reduce]`` table
with ``[reduce.umap]`` and ``[reduce.umap.kwargs]``.
"""

import tomlkit
from tomlkit.toml_document import TOMLDocument

from hyrax.config_migrations.migration_utils import migration_step, move_key


@migration_step(
    from_version=4,
    key_renames={
        "umap.fit_sample_size": "reduce.umap.fit_sample_size",
        "umap.model_path": "reduce.umap.model_path",
        "umap.save_fit_umap": "reduce.save_fit_model",
        "umap.parallel": "reduce.parallel",
        "umap.UMAP": "reduce.umap.kwargs",
    },
)
def move_umap_to_reduce(cfg: TOMLDocument) -> TOMLDocument:
    """Move the legacy ``[umap]`` and ``[umap.UMAP]`` to be under ``[reduce]``."""
    # Moving umap sections
    umap_tbl = cfg.get("umap")
    if not umap_tbl:
        return cfg

    # Ensure [reduce] exists
    reduce_tbl = cfg.get("reduce")
    if reduce_tbl is None:
        reduce_tbl = tomlkit.table()
        cfg["reduce"] = reduce_tbl

    # Ensure [reduce.umap] exists
    umap_reduce = reduce_tbl.get("umap")
    if umap_reduce is None:
        umap_reduce = tomlkit.table()
        reduce_tbl["umap"] = umap_reduce

    # under [reduce.umap]
    move_key(cfg, "umap.fit_sample_size", "reduce.umap.fit_sample_size")
    move_key(cfg, "umap.model_path", "reduce.umap.model_path")

    # under [reduce]
    reduce_tbl["batch_size"] = 1024
    move_key(cfg, "umap.save_fit_umap", "reduce.save_fit_model")
    move_key(cfg, "umap.parallel", "reduce.parallel")
    if "name" in umap_tbl and umap_tbl["name"] == "umap.UMAP":
        reduce_tbl["algorithm"] = "umap"

    # Move umap.UMAP kwargs to reduce.umap.kwargs
    move_key(cfg, "umap.UMAP", "reduce.umap.kwargs")

    # Delete the old umap section
    del cfg["umap"]

    # Adding tsne section
    reduce_tbl["tsne"] = tomlkit.table()

    reduce_tbl["tsne"]["kwargs"] = tomlkit.table()
    reduce_tbl["tsne"]["kwargs"]["n_components"] = 2
    reduce_tbl["tsne"]["kwargs"]["perplexity"] = 30.0

    # Adding pca section
    reduce_tbl["pca"] = tomlkit.table()
    reduce_tbl["pca"]["fit_sample_size"] = 1024
    reduce_tbl["pca"]["model_path"] = False

    reduce_tbl["pca"]["kwargs"] = tomlkit.table()
    reduce_tbl["pca"]["kwargs"]["n_components"] = 2

    if len(reduce_tbl):
        cfg["reduce"] = reduce_tbl

    return cfg
