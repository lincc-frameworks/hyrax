"""Splitting and dataset balancing utilities for Hyrax datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tomlkit

if TYPE_CHECKING:
    from hyrax.datasets.data_provider import DataProvider

logger = logging.getLogger(__name__)


# ── Private helpers ────────────────────────────────────────────────────────────


def _is_path_value(val: Any) -> bool:
    """Return True when val is a non-empty string (i.e. a file path, not a fraction)."""
    return isinstance(val, str) and bool(val)


def _resolve_seed(config: dict) -> int | None:
    """Return the effective RNG seed, resolving '' to config['data_set']['seed']."""
    rng_seed = config.get("split", {}).get("rng_seed", "")
    if not rng_seed:
        raw = config.get("data_set", {}).get("seed")
        return raw if raw else None
    return rng_seed


def _shuffle(indices: list[int], config: dict) -> None:
    """Shuffle *indices* in-place using the configured RNG.

    When ``split.rng_seed`` is empty, reproduces the legacy global-seed shuffle
    used by ``create_splits_from_fractions`` bit-for-bit.
    """
    rng_seed = config["split"]["rng_seed"]
    if not rng_seed:
        seed = config["data_set"]["seed"] if config["data_set"]["seed"] else None
        np.random.seed(seed)
        np.random.shuffle(indices)
    else:
        np.random.default_rng(rng_seed).shuffle(indices)


def _primary_instance(provider: DataProvider) -> Any:
    """Return the primary dataset instance from *provider*."""
    return provider.prepped_datasets[provider.primary_dataset]


def _find_primary_cfg(group_dict: dict) -> dict | None:
    """Return the first dataset config in *group_dict* that has primary_id_field set."""
    for cfg in group_dict.values():
        if isinstance(cfg, dict) and cfg.get("primary_id_field"):
            return cfg
    return None


def _compute_weights(
    indices: list[int],
    index_to_label: dict[int, Any],
    distribution: dict,
    num_classes: int,
) -> np.ndarray:
    """Compute per-sample WeightedRandomSampler weights.

    w_i = target_{class(i)} / count_{class(i)}  (raw, not normalised — WRS
    normalises internally, and raw values stay interpretable against distribution).
    """
    count_c: dict[Any, int] = {}
    for idx in indices:
        label = index_to_label[idx]
        count_c[label] = count_c.get(label, 0) + 1

    if distribution:
        target_c = {label: float(distribution.get(label, 0)) for label in count_c}
    else:
        uniform = 1.0 / num_classes
        target_c = {label: uniform for label in count_c}

    return np.array(
        [target_c.get(index_to_label[idx], 0) / count_c[index_to_label[idx]] for idx in indices],
        dtype=np.float64,
    )


# ── Validation ─────────────────────────────────────────────────────────────────


def validate_split_config(config: dict, datasets: dict[str, DataProvider]) -> None:
    """Validate ``[split]`` config values.

    Raises
    ------
    RuntimeError
        On any violated constraint (mixed float/path, bad domain, shared-location
        sum > 1.0, paths not in same directory).
    """
    split_cfg = config["split"]
    group_values = {name: split_cfg.get(name, 1.0) for name in datasets}

    path_groups = [n for n, v in group_values.items() if _is_path_value(v)]
    float_groups = [n for n in group_values if n not in path_groups]

    if path_groups and float_groups:
        raise RuntimeError(
            "split values must be all floats or all paths, not mixed. "
            f"Float groups: {float_groups}; path groups: {path_groups}."
        )

    if path_groups:
        parents = {Path(str(split_cfg[name])).parent for name in path_groups}
        if len(parents) > 1:
            raise RuntimeError(
                "All split path files must share a common parent directory. "
                f"Found multiple parents: {[str(p) for p in parents]}"
            )
        return

    # All floats: validate domain
    for name in datasets:
        raw = split_cfg.get(name, 1.0)
        frac = 1.0 if raw in ("", None, False) else float(raw)
        if not (0.0 < frac <= 1.0):
            raise RuntimeError(f"split.{name} = {frac} is out of range (0.0, 1.0].")

    # Sum check per shared primary_data_location (infer always independent)
    location_fracs: dict[str, list[float]] = {}
    for name, provider in datasets.items():
        if name == "infer":
            continue
        loc = provider.primary_data_location
        if loc:
            raw = split_cfg.get(name, 1.0)
            frac = 1.0 if raw in ("", None, False) else float(raw)
            location_fracs.setdefault(loc, []).append(frac)

    for loc, fracs in location_fracs.items():
        total = sum(fracs)
        if np.round(total, 5) > 1.0:
            raise RuntimeError(
                f"split fractions for data_location '{loc}' sum to {total:.6f}, which exceeds 1.0."
            )


def validate_balance_config(config: dict, datasets: dict[str, DataProvider]) -> None:
    """Validate ``[balance]`` config values (pre-scan checks only).

    Raises
    ------
    RuntimeError
        If getter is missing, distribution is malformed, or distribution sum ≠ 1.0.
    """
    balance_cfg = config["balance"]
    field = balance_cfg["field"] if balance_cfg["field"] else None
    balance_groups = balance_cfg["groups"]
    distribution = balance_cfg["distribution"]

    if not field:
        if balance_groups or distribution:
            raise RuntimeError(
                "balance.field must be set when balance.groups or balance.distribution are provided."
            )
        return
    for group_name, provider in datasets.items():
        if group_name == "infer":
            continue
        primary_ds = _primary_instance(provider)
        if not hasattr(primary_ds, f"get_{field}"):
            raise RuntimeError(
                f"balance.field='{field}' requires a get_{field} method on the primary "
                f"dataset of group '{group_name}', but none was found on "
                f"{type(primary_ds).__name__}."
            )

    for g in balance_groups:
        if g not in datasets:
            logger.warning("balance.groups contains '%s' which is not in data_request; ignoring.", g)

    if distribution:
        for label, val in distribution.items():
            try:
                fval = float(val)
            except (TypeError, ValueError) as err:
                raise RuntimeError(
                    f"balance.distribution['{label}'] = {val!r} is not a valid float."
                ) from err
            if not (0.0 < fval <= 1.0):
                raise RuntimeError(f"balance.distribution['{label}'] = {fval} is out of range (0.0, 1.0].")
        total = sum(float(v) for v in distribution.values())
        if np.round(total, 5) != 1.0:
            raise RuntimeError(
                f"balance.distribution values sum to {total:.6f}; they must sum to exactly 1.0."
            )

    # [label] pre-scan checks (only consulted when both label table and distribution are present)
    label_cfg = config["label"]
    if label_cfg:
        raw_values = list(label_cfg.values())
        if len(raw_values) != len(set(str(v) for v in raw_values)):
            raise RuntimeError(
                "[label] values must be unique — two or more aliases map to the same raw value."
            )
        if distribution:
            for dist_key in distribution:
                if dist_key not in label_cfg:
                    raise RuntimeError(
                        f"balance.distribution key '{dist_key}' is not defined in [label]. "
                        "All distribution keys must appear in [label] when [label] is non-empty."
                    )


def validate_distribution_labels(distribution: dict, observed_labels: set) -> None:
    """Cross-check distribution keys against the observed class labels (post-scan).

    Raises
    ------
    RuntimeError
        If distribution contains a label absent from the dataset.
    """
    if not distribution:
        return
    for label in distribution:
        if label not in observed_labels:
            raise RuntimeError(
                f"balance.distribution contains label '{label}' not found in the dataset. "
                f"Observed labels: {sorted(observed_labels)}"
            )
    for label in observed_labels:
        if label not in distribution:
            logger.warning(
                "Dataset class '%s' is absent from balance.distribution; "
                "it will receive weight 0 (no samples drawn for this class).",
                label,
            )


# ── Core split computation ─────────────────────────────────────────────────────


def _compute_splits(config: dict, datasets: dict[str, DataProvider]) -> dict[str, dict]:
    """Compute split indices (and optional balance weights) for each group.

    Returns
    -------
    dict mapping group_name → {"indexes": np.ndarray[int64], "weights": np.ndarray[float64] | None}
    """
    split_cfg = config["split"]
    balance_cfg = config["balance"]
    field = balance_cfg["field"] if balance_cfg["field"] else None
    balance_groups_cfg = balance_cfg["groups"]
    distribution = balance_cfg["distribution"]

    # Resolve groups_to_balance per spec §4.2 table
    if balance_groups_cfg:
        groups_to_balance = set(balance_groups_cfg)
    elif distribution and field:
        groups_to_balance = set(datasets.keys()) - {"infer"}
    else:
        groups_to_balance = set()

    result: dict[str, dict] = {}

    # Infer: always independent, no shuffle, no weights
    if "infer" in datasets:
        provider = datasets["infer"]
        n_items = len(provider)
        frac = float(split_cfg.get("infer", 1.0)) if not _is_path_value(split_cfg.get("infer")) else 1.0
        count = round(n_items * frac)
        result["infer"] = {
            "indexes": np.array(list(range(count)), dtype=np.int64),
            "weights": None,
        }

    # Group remaining providers by primary_data_location
    non_infer = {k: v for k, v in datasets.items() if k != "infer"}
    location_groups: dict[str, dict[str, DataProvider]] = {}
    for group_name, provider in non_infer.items():
        loc = provider.primary_data_location or group_name
        location_groups.setdefault(loc, {})[group_name] = provider

    for _loc, loc_datasets in location_groups.items():
        first_provider = next(iter(loc_datasets.values()))
        n_items = len(first_provider)
        fractions = {name: float(split_cfg.get(name, 1.0)) for name in loc_datasets}
        total = sum(fractions.values())
        last_name = list(loc_datasets.keys())[-1]

        if not field:
            # Non-stratified: mirror create_splits_from_fractions semantics
            indices = list(range(n_items))
            _shuffle(indices, config)

            offset = 0
            for name, frac in fractions.items():
                count = min(round(n_items * frac), n_items - offset)
                if name == last_name and total >= 1.0 - 1e-5:
                    count = n_items - offset
                result[name] = {
                    "indexes": np.array(indices[offset : offset + count], dtype=np.int64),
                    "weights": None,
                }
                offset += count

        else:
            # Stratified: build class index map, then distribute per-class
            primary_ds = _primary_instance(first_provider)
            getter = getattr(primary_ds, f"get_{field}")

            class_inds: dict[Any, list[int]] = {}
            for i in range(n_items):
                label = getter(i)
                class_inds.setdefault(label, []).append(i)

            # [label] re-keying: translate raw values to alias strings (§4.3)
            label_cfg = dict(config.get("label") or {})
            if label_cfg:
                raw_to_name = {v: k for k, v in label_cfg.items()}
                rekeyed: dict[Any, list[int]] = {}
                for raw_val, inds in class_inds.items():
                    alias = raw_to_name.get(raw_val)
                    if alias is None:
                        logger.warning(
                            "Dataset contains raw label value %r from get_%s "
                            "that has no alias in [label]; %d item(s) with this value "
                            "will be excluded from all split groups.",
                            raw_val,
                            field,
                            len(inds),
                        )
                    else:
                        rekeyed[alias] = inds
                class_inds = rekeyed

            validate_distribution_labels(distribution, set(class_inds))

            # Build reverse lookup for weight computation
            index_to_label: dict[int, Any] = {}
            for label, inds in class_inds.items():
                for i in inds:
                    index_to_label[i] = label

            num_classes = len(class_inds)
            per_group: dict[str, list[int]] = {name: [] for name in loc_datasets}

            for label in sorted(class_inds, key=str):
                inds = list(class_inds[label])
                _shuffle(inds, config)
                offset = 0
                for name, frac in fractions.items():
                    count = min(round(len(inds) * frac), len(inds) - offset)
                    per_group[name] += inds[offset : offset + count]
                    offset += count
                if offset < len(inds) and total >= 1.0 - 1e-5:
                    per_group[last_name] += inds[offset:]

            for name, indices_list in per_group.items():
                if name in groups_to_balance:
                    weights = _compute_weights(indices_list, index_to_label, distribution, num_classes)
                else:
                    weights = None
                result[name] = {
                    "indexes": np.array(indices_list, dtype=np.int64),
                    "weights": weights,
                }

    return result


# ── Persistence / loading ──────────────────────────────────────────────────────


def persist_splits(results_dir: Path, splits: dict[str, dict], config: dict) -> None:
    """Write one ``<group>_split.npz`` per group and a ``split_config.toml``.

    The ``weights`` array is omitted entirely for unbalanced groups (``None``)
    to save space; ``load_split_files`` treats its absence as ``None``.
    """
    for group, data in splits.items():
        save_kwargs: dict[str, np.ndarray] = {"indexes": data["indexes"]}
        if data["weights"] is not None:
            save_kwargs["weights"] = data["weights"]
        np.savez_compressed(results_dir / f"{group}_split.npz", **save_kwargs)

    split_config: dict = {}
    for key in ("data_request", "split", "balance", "label"):
        if key in config:
            split_config[key] = config[key]
    with open(results_dir / "split_config.toml", "w") as f:
        f.write(tomlkit.dumps(split_config))


def load_split_files(paths: dict[str, Path]) -> dict[str, dict]:
    """Load previously persisted split files.

    Parameters
    ----------
    paths:
        Mapping of group name → path to ``<group>_split.npz``.

    Returns
    -------
    dict mapping group_name → {"indexes": ndarray, "weights": ndarray | None}
    """
    result: dict[str, dict] = {}
    for group, path in paths.items():
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"Split file for group '{group}' not found: {path}")
        npz = np.load(path)
        if "indexes" not in npz.files:
            raise RuntimeError(f"Split file '{path}' is missing the required 'indexes' array.")
        result[group] = {
            "indexes": npz["indexes"],
            "weights": npz["weights"] if "weights" in npz.files else None,
        }
    return result


def assign_splits_to_providers(datasets: dict[str, DataProvider], splits: dict[str, dict]) -> None:
    """Attach split indices and weights onto each provider in *datasets*."""
    for group, data in splits.items():
        if group not in datasets:
            continue
        provider = datasets[group]
        provider.split_indices = data["indexes"].tolist()
        provider.split_weights = data["weights"]


# ── Equivalency ────────────────────────────────────────────────────────────────


def configs_equivalent(prev: dict, cur: dict) -> tuple[bool, list[str]]:
    """Check whether *prev* config would produce the same splits as *cur*.

    Returns
    -------
    (equivalent, diffs)
        *equivalent* is True only when all compared fields match.
        *diffs* is a human-readable list of differences (empty when equivalent).
    """
    diffs: list[str] = []

    def _get(cfg: dict, *keys: str, default: Any = None) -> Any:
        node = cfg
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node

    # Global comparisons
    prev_field = _get(prev, "balance", "field") or ""
    cur_field = _get(cur, "balance", "field") or ""
    if str(prev_field) != str(cur_field):
        diffs.append(f"balance.field: {prev_field!r} → {cur_field!r}")

    prev_dist = dict(_get(prev, "balance", "distribution") or {})
    cur_dist = dict(_get(cur, "balance", "distribution") or {})
    if {str(k): float(v) for k, v in prev_dist.items()} != {str(k): float(v) for k, v in cur_dist.items()}:
        diffs.append("balance.distribution changed")

    if _resolve_seed(prev) != _resolve_seed(cur):
        diffs.append(f"rng_seed (resolved): {_resolve_seed(prev)!r} → {_resolve_seed(cur)!r}")

    # Per-group comparisons
    cur_dr = _get(cur, "data_request") or {}
    prev_dr = _get(prev, "data_request") or {}

    for group_name in cur_dr:
        if group_name not in prev_dr:
            diffs.append(f"group '{group_name}' absent from previous split config")
            continue

        cur_primary = _find_primary_cfg(cur_dr[group_name])
        prev_primary = _find_primary_cfg(prev_dr[group_name])

        if cur_primary is None or prev_primary is None:
            diffs.append(f"group '{group_name}': cannot find primary dataset config")
            continue

        if cur_primary.get("dataset_class") != prev_primary.get("dataset_class"):
            diffs.append(
                f"group '{group_name}' dataset_class: "
                f"{prev_primary.get('dataset_class')!r} → {cur_primary.get('dataset_class')!r}"
            )

        if cur_primary.get("data_location") != prev_primary.get("data_location"):
            diffs.append(
                f"group '{group_name}' data_location: "
                f"{prev_primary.get('data_location')!r} → {cur_primary.get('data_location')!r}"
            )

        cur_frac = _get(cur, "split", group_name)
        prev_frac = _get(prev, "split", group_name)
        if cur_frac != prev_frac:
            diffs.append(f"split.{group_name}: {prev_frac!r} → {cur_frac!r}")

        cur_groups = list(_get(cur, "balance", "groups") or [])
        prev_groups = list(_get(prev, "balance", "groups") or [])
        cur_in = group_name in cur_groups
        prev_in = group_name in prev_groups
        if cur_in != prev_in:
            diffs.append(f"group '{group_name}' balance.groups membership: {prev_in} → {cur_in}")

    return (len(diffs) == 0, diffs)


def find_equivalent_split(config: dict, results_root: Path | None = None) -> dict[str, Path] | None:
    """Scan the results directory for a previously persisted equivalent split.

    Returns the group→npz path mapping of the first match, or ``None``.
    """
    if results_root is None:
        results_root = Path(config["general"]["results_dir"]).expanduser().resolve()

    if not results_root.exists():
        return None

    split_dirs = sorted(
        (p for p in results_root.glob("*-splits-*") if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )

    for split_dir in split_dirs:
        config_path = split_dir / "split_config.toml"
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                prev_config = dict(tomlkit.parse(f.read()))
        except Exception:
            continue

        equivalent, _ = configs_equivalent(prev_config, config)
        if not equivalent:
            continue

        paths: dict[str, Path] = {}
        for group in config.get("data_request") or {}:
            npz_path = split_dir / f"{group}_split.npz"
            if npz_path.exists():
                paths[group] = npz_path
        if paths:
            return paths

    return None


# ── Public driver ──────────────────────────────────────────────────────────────


def create_splits(
    config: dict,
    datasets: dict[str, DataProvider],
    *,
    results_dir: Path | None = None,
    persist: bool = True,
) -> dict[str, dict]:
    """Compute (or load) splits and weights for each data group.

    Assigns ``split_indices`` / ``split_weights`` on each provider via
    :func:`assign_splits_to_providers`.

    Returns
    -------
    dict mapping group_name → {"indexes": ndarray[int64], "weights": ndarray[float64] | None}
    """
    validate_split_config(config, datasets)
    validate_balance_config(config, datasets)

    split_cfg = config.get("split", {})

    # Determine whether paths were supplied
    using_paths = any(_is_path_value(split_cfg.get(name)) for name in datasets)

    if using_paths:
        paths = {name: Path(str(split_cfg[name])) for name in datasets if name in split_cfg}
        splits = load_split_files(paths)

        # Warn if the sibling split_config.toml differs from current config
        first_path = next(iter(paths.values()))
        sibling_cfg_path = first_path.parent / "split_config.toml"
        if sibling_cfg_path.exists():
            try:
                with open(sibling_cfg_path) as f:
                    prev_config = dict(tomlkit.parse(f.read()))
                equivalent, diffs = configs_equivalent(prev_config, config)
                if not equivalent:
                    logger.warning(
                        "Supplied split files were produced with a different config. Differences: %s",
                        "; ".join(diffs),
                    )
            except Exception:
                pass

        assign_splits_to_providers(datasets, splits)
        return splits

    # Search for a reusable equivalent split
    equivalent_paths = find_equivalent_split(config)
    if equivalent_paths is not None:
        logger.info("Reusing equivalent split from %s", next(iter(equivalent_paths.values())).parent)
        splits = load_split_files(equivalent_paths)
        assign_splits_to_providers(datasets, splits)
        if persist and results_dir is not None:
            persist_splits(results_dir, splits, config)
        return splits

    # Compute fresh splits
    splits = _compute_splits(config, datasets)
    assign_splits_to_providers(datasets, splits)

    if persist and results_dir is not None:
        persist_splits(results_dir, splits, config)

    return splits
