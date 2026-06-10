# Splits and Dataset Balancing — Technical Specification

Implementation spec derived from the design doc *"Splits and dataset balancing in
Hyrax."* This document is the implementation-facing companion: it resolves the
ambiguities in the design, names concrete modules/functions/config keys, and
describes the algorithms precisely enough to code from. Augmentation is covered in
a separate design doc and is **out of scope** here except where it touches the
config surface (noted inline).

---

## 1. Overview

Add configuration-driven, reproducible dataset **splitting** and class
**rebalancing** to Hyrax:

- Two new top-level TOML tables: `[split]` (per-group fractions or paths + seed)
  and `[balance]` (class field, which groups to balance, target distribution).
- A new class-based verb, `create_splits` (callable as `h.create_splits()`),
  thin wrapper over a new utility module `splitting_utils.py`.
- `split_fraction` moves out of `[data_request]` into `[split]` (config migration).
- `setup_dataset` stops computing splits; it returns only
  `{group_name: DataProvider}`. All split/balance logic lives in
  `splitting_utils.py`.
- Splits are persisted to a timestamped results directory and can be **reused**
  via an equivalency check, avoiding repeated full data scans.
- `dist_data_loader` gains `WeightedRandomSampler` + `torch.utils.data.Subset`
  support so balancing weights drive sampling without materializing
  full-length weight arrays.

The default behavior (no `[split]` edits, no `[balance]` edits) **must reproduce
current Hyrax behavior**.

---

## 2. Goals / Non-goals

**Goals**

1. Easy things easy: untouched `[split]`/`[balance]` ⇒ current behavior.
2. Progressive: `[split]` alone for custom fractions; add `[balance]` for
   stratification/rebalancing.
3. Reproducible & shareable: split index files + the config that produced them
   are persisted; a config-equivalency check enables reuse.
4. Scales to O(10–100M) items: never allocate a full-length weight array when a
   subset is selected (use `Subset` + per-subset weights).
5. Backward compatible for train/validation and for inference subset selection.

**Non-goals (this spec)**

- Data augmentation (`augment_<field>` methods, TTA) — separate doc. We only
  reserve the warning/error hooks described in §13.
- A dataset-content hash for equivalency (future; we compare config only).
- Removing the legacy `[data_set].{train,validate,test}_size` path and the
  deprecated `pytorch_ignite.create_splits(dataset, config)` function — that
  removal lands separately in **PR #944**. This spec assumes the **post-#944
  codebase**: no `split` parameter on `dist_data_loader`, no legacy multi-split
  path, legacy `[data_set]` size keys hard-error in `setup_dataset` (§11, §14).

---

## 3. Decisions that resolve design-doc ambiguities

The design doc is internally inconsistent in a few places. This spec fixes the
following and these decisions are **normative**:

| # | Design doc | This spec uses | Rationale |
|---|-----------|----------------|-----------|
| D1 | `[split]` vs `h.set_config("splits", …)` | Table name **`split`** (singular); verb **`create_splits`** (plural) | `split`/`balance` match the Configuration section; the `"splits"` setter calls are typos. |
| D2 | `validation = 1.0` in `[split]` defaults; pseudocode uses `validate` | Group name **`validate`** everywhere | Matches existing code (`Train.OPTIONAL_DATA_GROUPS = ("validate", "test")`) and all data_request groups. |
| D3 | In-memory arrays called `indexes`/`weights`, `split_indexes`/`split_weights`, `split_indices`/`split_weights` | DataProvider attrs **`split_indices`** (already exists) and new **`split_weights`** | `split_indices` is the existing attribute (`data_provider.py:318`). |
| D4 | Equivalency lists `balance.label` | **`balance.field`** | `balance.field` is the actual config key; `balance.label` is a typo. |
| D5 | Results dir `…/<datetime>_splits_<unique_string>` | `create_results_dir(config, "splits")` ⇒ `YYYYMMDD-HHMMSS-splits-<uid>` | Reuse the existing helper and house naming convention (`config_utils.py:622`). |
| D6 | `make_sampler` always returns `WeightedRandomSampler` when shuffling | WRS **only when balancing weights are present**; uniform case keeps `SubsetRandomSampler` (no replacement) | WRS uses `replacement=True`; with uniform weights that would silently change default training from a per-epoch permutation to a bootstrap sample. See §11. |
| D7 | `make_sampler(...)` sequential branch returns `SubsetSequentialSampler(indexes)` while also wrapping in `Subset(dataset, indexes)` | Sampler operates in **subset-local** index space `range(len(subset))` | Once wrapped in `Subset`, indices are local; using original indices would double-index. |
| D8 | `dist_data_loader` returns a 3-tuple `(loader, indexes, weights)` | Return **the `DataLoader` only** | Every callsite today discards the second element (`x, _ = …`); indexes and weights already ride on the DataProvider (`split_indices`/`split_weights`). Callsite updates listed in §12. |
| D9 | Pseudocode sets `weights[label] ← len(class_inds[label]) / len(dataset)` (class frequency) and then `weights[label] ← balance.distribution[label]` (raw target) | **`w_i = target_{class(i)} / count_{class(i)}`** (§5) | The pseudocode's raw values do not produce the requested class distribution under `WeightedRandomSampler`; the design's own prose ("the weights … will be the inverse frequency of the class label") matches §5. |
| D10 | "Design of create_splits" says paths *use* the equivalency check; "Defining equivalency" says **no** equivalence check for paths | Paths ⇒ **no search** for equivalent splits, but **do** compare the sibling `split_config.toml` to the current config and log a warning listing differences (§7.1 step 2) | Honors both passages: user-supplied files are always used (trusted), and the user is told when their current config would not have produced them. |

---

## 4. Configuration schema

### 4.1 `[split]` table

```toml
[split]
train    = 1.0   # float in (0, 1]  OR  a path to a previously generated split file
validate = 1.0
test     = 1.0
infer    = 1.0
rng_seed = ""    # "" ⇒ fall back to config["data_set"]["seed"]
```

Semantics:

- **`split.<group>`** — fraction of the group's primary dataset to use, OR a
  filesystem path (string) to a previously generated `<group>_split.npz`.
  - Float domain: `0.0 < f <= 1.0`. Default `1.0`.
  - For groups that share a `data_location` (e.g. `train`+`validate` over the
    same source), `0.0 < Σ fractions <= 1.0`; the splits are **non-overlapping**
    partitions of that shared source. Σ < 1.0 is allowed (use a subset).
  - For groups with distinct `data_location`s, each fraction is an independent
    subset of that group's own data.
  - `infer` is always treated independently (never partitioned against
    train/validate/test).
- **`split.rng_seed`** — RNG seed for shuffling/partitioning.
  - `""` (default) ⇒ fall back to **exactly the RNG mechanism the current code
    uses**: the global NumPy RNG seeded with `config["data_set"]["seed"]`
    (`false` ⇒ `None` ⇒ system entropy):

    ```python
    seed = config["data_set"]["seed"] if config["data_set"]["seed"] else None
    np.random.seed(seed)
    np.random.shuffle(indices)
    ```

    This reproduces today's `create_splits_from_fractions` shuffle
    (`pytorch_ignite.py:485–487`) bit-for-bit, so a default config yields
    splits identical to current Hyrax.
  - A non-empty `rng_seed` ⇒ shuffle with a dedicated
    `np.random.default_rng(rng_seed)` instead (does not touch global RNG state).

**Defaults & unknown group keys.** The four standard groups
(`train`/`validate`/`test`/`infer`) plus `rng_seed` are shipped in
`hyrax_default_config.toml` so no validation warning fires for them. A user may
add a non-standard group key (e.g. `split.finetune`). Today
`ConfigManager._validate_runtime_config` (`config_utils.py:505`) only **warns**
on keys without a default — it does not error — so non-standard groups still work.
To suppress that warning for legitimately dynamic tables, add `split` and
`balance.distribution` to a new `DYNAMIC_KEY_TABLES` allowlist consulted by
`_validate_runtime_config` (see §6.3). This is **not** Pydantic validation.

**Mixing fractions and paths (validated in `create_splits`):**

- All `split.<group>` values must be **either all floats or all paths** — mixing
  raises `RuntimeError`.
- If paths: every path's **parent directory must be identical** (they should be
  the files of a single persisted split dir); otherwise raise.
- When paths are supplied, no fresh split is computed and no **search** for
  equivalent splits is performed — the user is trusted. If a
  `split_config.toml` sits next to the supplied files, it is compared against
  the current config and any differences are **logged as a warning** (never an
  error). See §3 D10 and §7.1 step 2.

### 4.2 `[balance]` table

```toml
[balance]
field  = ""   # class field name on the primary dataset; "" ⇒ no stratification/rebalancing
groups = []   # data groups to rebalance, e.g. ["train"]; [] ⇒ no rebalancing

[balance.distribution]
# empty ⇒ equal target distribution across all observed classes
# label_0 = 0.25
# label_1 = 0.75
```

Semantics:

- **`balance.field`** — name of the field identifying each item's class. Must
  resolve to a `get_<field>` getter on the **primary** dataset of each relevant
  group; otherwise raise `RuntimeError`. When set, splits become **stratified**
  (each split preserves per-class proportions). When `""`/falsy, behavior is the
  current non-stratified shuffle-and-slice.
- **`balance.groups`** — the groups whose sampling weights are adjusted to the
  target distribution. Default `[]` (explicit opt-in; no surprise rebalancing).
  - If `groups` names a group **absent** from `data_request`: **warn** (extra
    group ignored).
  - Groups present in `data_request` but not in `groups` keep their **natural**
    class distribution (weights `None`, §5). This is expected.
  - Setting `groups` is independent of stratification: `field` set + `groups`
    empty + `distribution` empty ⇒ stratified splits only, no reweighting.
  - **Which groups get rebalanced** (normative; encodes the design doc's "Use
    of the keys in the balance table", requires `balance.field` to be set):

    | `balance.groups` | `balance.distribution` | Groups rebalanced |
    |---|---|---|
    | empty | empty | none — stratified splits only |
    | non-empty | empty | the listed groups, to a **uniform** target |
    | empty | non-empty | **all** groups in `data_request` (except `infer`), to the given target |
    | non-empty | non-empty | the listed groups, to the given target |

    Call the resolved set `groups_to_balance`; §5 and §7.3 are written against
    it. Compute it once, e.g.
    `groups_to_balance = set(balance.groups) or (set(data_request) - {"infer"} if balance.distribution else set())`.
- **`balance.distribution`** — maps class label → target fraction in
  `(0.0, 1.0]`. Validation:
  - Values must sum to **exactly 1.0** (use `np.round(Σ, 5) == 1.0`); else raise.
  - A label not present in the dataset ⇒ raise `RuntimeError`.
  - A dataset class missing from `distribution` ⇒ assume weight 0 and **warn**.
  - Empty ⇒ equal target across all observed classes (uniform).

> **Note (`key = false` ⇒ None).** Per Hyrax convention, `field = false` and
> `field = ""` both mean "unset". Code must treat falsy `field`/`groups`/
> `distribution` as "not configured".

---

## 5. Weighting model (precise formulas)

Weights are the per-sample weights handed to `WeightedRandomSampler` for a given
split. Let a split contain indices `I`, with `class(i)` the label of item `i`,
`count_c = |{i ∈ I : class(i) = c}|`, and `target_c` the target fraction for
class `c`.

- **No balancing** (`balance.field` falsy, or group ∉ `groups_to_balance`,
  §4.2): `weights = None`.

  **Why `None` rather than an array of `1.0`** (normative):
  1. `weights is None` keeps the uniform case on `SubsetRandomSampler` — a
     per-epoch permutation **without replacement**, byte-for-byte today's
     default training behavior. An all-`1.0` array would route through
     `WeightedRandomSampler(replacement=True)` (§11), silently turning every
     epoch into a bootstrap sample (duplicates and omissions within an epoch).
  2. Avoids allocating an O(N) float array per unbalanced group (matters at
     Rubin scale).
  3. `None` is an unambiguous "not balanced" sentinel for `__repr__` (§8),
     persistence (the `weights` array is omitted from the `.npz`, §7.4), and
     equivalency reporting.

- **Balancing (group ∈ `groups_to_balance`), no explicit distribution**
  (uniform target, `K` observed classes):
  `target_c = 1/K`, so `w_i = (1/K) / count_{class(i)}  ∝  1 / count_{class(i)}`
  — the inverse class frequency. (Equal expected draws per class under
  replacement sampling.)

- **Balancing (group ∈ `groups_to_balance`) with explicit `distribution`**:
  `w_i = target_{class(i)} / count_{class(i)}`.
  Classes with `target_c = 0` (missing from distribution) get `w_i = 0`.

Store the **raw** `target_c / count_c` values — do **not** normalize (decided in
review). WRS normalizes internally so there is no functional difference, and raw
values keep the persisted weights directly interpretable against the configured
distribution. `num_samples` for WRS is `len(I)`.

`infer` always uses `weights = None` (no rebalancing ever).

---

## 6. Config migration & schema changes

### 6.1 Migration `003_move_split_fraction_to_split.py`

`split_fraction` currently lives per-dataset inside `[data_request]`. It moves to
`[split].<group>`. This is a flatten (per-group primary's `split_fraction` ⇒
`split[group]`), so it needs a **custom function body** — `key_renames` cannot
express it (leave `key_renames={}` or omit; it is only used for runtime
deprecation warnings).

```python
# src/hyrax/config_migrations/migrations/003_move_split_fraction_to_split.py
import tomlkit
from tomlkit.toml_document import TOMLDocument
from hyrax.config_migrations.migration_utils import migration_step

@migration_step(from_version=3)
def move_split_fraction_to_split(cfg: TOMLDocument) -> TOMLDocument:
    """Move per-dataset `split_fraction` from [data_request] into [split]."""
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
```

- Registering `from_version=3` auto-derives `CURRENT_CONFIG_VERSION = 4`
  (currently 3). Do **not** hand-edit the version or the default config's
  stamped version.
- Add tests to `tests/hyrax/test_config_migrations.py` covering: (a) a legacy
  v3 config with `split_fraction` migrates to `[split]` and strips the old key;
  (b) a clean v4 config is a no-op.

### 6.2 Pydantic `DataRequestConfig` changes (`config_schemas/data_request.py`)

- **Remove** the `split_fraction` field.
- **Remove** the `require_primary_id_for_split_fraction` validator.
- **Remove** `DataRequestDefinition.validate_cross_group` split_fraction logic
  (the sum-≤1.0 and consistency checks). These checks move to
  `splitting_utils.validate_split_config` (§7.2), which can see `[split]`. The
  `Verb.validate_data_request` call to `validate_cross_group` becomes a no-op or
  is removed; replace its responsibility with the split-config validation invoked
  by the `create_splits` driver.
- `DataProvider.__repr__` currently prints `split_fraction` from the request
  dict — update it to read split info from the attached `split_indices`/
  `split_weights` instead (§8).

> This is consistent with CLAUDE.md: **do not add** new Pydantic validation to
> `[split]`/`[balance]`. They are validated by code in `splitting_utils.py`.

### 6.3 `DYNAMIC_KEY_TABLES` (optional warning suppression)

In `config_utils.py`, add a small allowlist used by `_validate_runtime_config`
to skip the "no default defined" warning for child keys of dynamic tables:

```python
# Tables whose *child keys* are user-defined (group names, class labels) and so
# legitimately have no entry in the default config.
DYNAMIC_KEY_TABLES = ("split", "distribution")  # 'distribution' nests under 'balance'
```

When recursing into a table whose name is in `DYNAMIC_KEY_TABLES`, do not warn on
unknown children. This is purely cosmetic (the current code only warns, never
errors), so it may be deferred, but is recommended to keep logs clean for
`balance.distribution` (arbitrary labels) and non-standard split groups.

---

## 7. New module: `src/hyrax/splitting_utils.py`

Houses all splitting, balancing, validation, persistence, and equivalency logic.
This is the "set of utility functions" the design references; the `create_splits`
verb is a thin wrapper over `create_splits()` here.

### 7.1 Public driver

```python
def create_splits(
    config: dict,
    datasets: dict[str, DataProvider],
    *,
    results_dir: Path | None = None,
    persist: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute (or load) splits + weights for each data group.

    Returns {group: {"indexes": np.ndarray[int64], "weights": np.ndarray[float64] | None}}.
    When persist and results_dir are set, also writes <group>_split.npz files and
    split_config.toml. Assigns split_indices/split_weights onto each provider via
    assign_splits_to_providers().
    """
```

Driver flow:

1. `validate_split_config(config, datasets)` and
   `validate_balance_config(config, datasets)` (§7.2). These are **pre-scan**
   checks only; the label-level distribution checks run later, inside §7.3,
   because they need the observed class labels from the full dataset scan.
2. If `split.*` values are **paths** ⇒ `load_split_files(paths)` (§7.4); attach
   to providers; return (no persistence). Do **not** search for equivalent
   splits — the user-supplied files are always used. But if a
   `split_config.toml` sits in the same directory as the supplied files,
   compare it to the current config with `configs_equivalent` and **log a
   warning listing the differences**; never raise (§3 D10).
3. Else ⇒ `find_equivalent_split(config)` (§7.5); if an equivalent persisted
   split is found, load those files and return (skips the full data scan).
4. Else compute splits (§7.3), assign to providers, persist if requested,
   return.

> Naming: the legacy `pytorch_ignite.create_splits(data_set, config)` is
> removed by PR #944, so `hyrax.splitting_utils.create_splits(config, datasets)`
> is the only symbol with this name.

### 7.2 Validation helpers

```python
def validate_split_config(config, datasets) -> None
def validate_balance_config(config, datasets) -> None
```

`validate_split_config` enforces:
- per-group float domain `(0,1]`;
- all-floats-or-all-paths (else raise);
- paths share a parent dir (else raise);
- for groups sharing a `primary_data_location`, `Σ fractions <= 1.0` (else raise).

`validate_balance_config` enforces the **pre-scan** §4.2 rules — everything
knowable from the config and dataset classes alone, before any data is read:
`field` getter exists on each relevant group's primary dataset;
`groups ⊆ data_request` (warn on extras); `distribution` values each in
`(0.0, 1.0]` and summing to exactly 1.0.

The **label-level** checks need the set of observed class labels and therefore
can only run **after** the full dataset scan. They live in a separate helper,
called from the stratified branch of §7.3 once `class_inds` has been built:

```python
def validate_distribution_labels(distribution: dict, observed_labels: set) -> None
```

- A `distribution` key not in `observed_labels` ⇒ raise `RuntimeError`.
- An observed label missing from a non-empty `distribution` ⇒ **warn** and
  treat its target as 0 — samples of that class get weight 0 (§5).
- No-op when `distribution` is empty.

### 7.3 Core split computation

Group the requested `datasets` by `provider.primary_data_location`. Choose the
RNG per §4.1 and wrap it in one helper so both branches below share it:

```python
def _shuffle(indices: list[int], config: dict) -> None:   # in-place
    rng_seed = config["split"]["rng_seed"]
    if rng_seed == "":   # fall back to the RNG the current code uses
        seed = config["data_set"]["seed"] if config["data_set"]["seed"] else None
        np.random.seed(seed)
        np.random.shuffle(indices)
    else:                # dedicated generator, global state untouched
        np.random.default_rng(rng_seed).shuffle(indices)
```

For each **location group** `G` (set of `(group_name, provider)` sharing a
source):

- **`infer` special case** (handled per-group, never partitioned): take the
  first `round(N * fraction)` indices of `range(N)` **without shuffling**;
  `weights = None`.

- **Non-stratified** (`balance.field` falsy): reproduce
  `create_splits_from_fractions` semantics — shuffle `range(N)` with
  `_shuffle`, slice contiguous non-overlapping blocks per group fraction (in
  deterministic group order), assign any remainder to the last block iff
  `Σ ≈ 1.0`. `weights = None` for all.

- **Stratified** (`balance.field` set): build `class_inds: {label: list[int]}`
  by scanning the primary dataset's `get_<field>` (full scan, O(N)), then call
  `validate_distribution_labels(balance.distribution, set(class_inds))` (§7.2).
  For each class, shuffle its indices and distribute **non-overlapping** across
  the groups in `G` by their fractions (same slicing logic, per class). This
  yields per-group indices that preserve class proportions. Then per group:
  - if `group_name ∈ groups_to_balance` (§4.2 — note this includes the
    `distribution` non-empty + `groups` empty case, where **every** non-infer
    group is balanced): compute `weights` per §5, with `count_c` measured
    **within that split**;
  - else: `weights = None`.

  Pseudocode for the per-class partitioning. Deterministic ordering: classes
  in sorted-label order, groups in `data_request` insertion order (matching
  the non-stratified branch):

  ```text
  per_group_indices = {g: [] for g in G}
  total = sum(fractions.values())
  for label in sorted(class_inds):
      inds = class_inds[label]
      _shuffle(inds, config)
      offset = 0
      for g, frac in fractions.items():            # data_request order
          count = min(round(len(inds) * frac), len(inds) - offset)
          per_group_indices[g] += inds[offset : offset + count]
          offset += count
      if offset < len(inds) and total >= 1.0 - 1e-5:
          per_group_indices[last group] += inds[offset:]   # remainder
  ```

Return `{group: {"indexes": ndarray, "weights": ndarray|None}}`.

### 7.4 Persistence / loading

```python
def persist_splits(results_dir: Path, splits, config) -> None
def load_split_files(paths: dict[str, Path]) -> dict[str, dict[str, np.ndarray]]
def assign_splits_to_providers(datasets, splits) -> None
```

- `persist_splits` writes one `<group>_split.npz` per group via
  `np.savez_compressed`: always an `indexes` array (`int64`); a `weights` array
  (`float64`) **only when the group's weights are not `None`**. When a group is
  unbalanced the `weights` key is **omitted entirely** (decided in review:
  saves space, and a by-hand file needs only `indexes`). Also writes
  `split_config.toml` containing `data_request`, `split`, and `balance` (via
  tomlkit / `log_runtime_config`-style dump). `.npz` gives compression at
  Rubin scale.
- `load_split_files(paths)` — `paths` is `{group: Path}`, built from the
  `split.<group>` config values. For each group:
  1. Raise `RuntimeError` if the file does not exist.
  2. `npz = np.load(path)`; raise `RuntimeError` if `"indexes"` is not in
     `npz.files`.
  3. `weights = npz["weights"] if "weights" in npz.files else None` —
     mirroring the `persist_splits` convention above.

  Returns `{group: {"indexes": ndarray, "weights": ndarray | None}}` — the
  same shape the compute path (§7.3) produces.
- `assign_splits_to_providers` sets `provider.split_indices = indexes.tolist()`
  (keep list type for existing samplers) and `provider.split_weights =
  weights` (`ndarray` or `None`).

### 7.5 Equivalency

```python
def find_equivalent_split(config, results_root: Path | None = None) -> dict[str, Path] | None
def configs_equivalent(prev: dict, cur: dict) -> tuple[bool, list[str]]
```

`find_equivalent_split` scans `results_dir/*-splits-*/split_config.toml`
(most-recent first), calls `configs_equivalent`, and returns the group→npz path
map of the first match (or `None`).

`configs_equivalent` returns `(True, diffs)` only if **all** of the following
hold (per the design):

Global (must be equal):
- `balance.field`
- `balance.distribution`
- `split.rng_seed` (compared as the *resolved* seed)

Per data group present in the **current** `data_request`:
- `data_request.<group>.<primary>.dataset_class`
- `data_request.<group>.<primary>.data_location`
- `split.<group>`
- membership parity: group ∈ `balance.groups` in prev **iff** group ∈
  `balance.groups` in cur.

`diffs` is a human-readable list used for the §7.1-step-2 warning when paths are
explicitly requested but the producing config differs. (We **cannot** compare an
in-memory loaded split against the current config beyond the persisted TOML —
the user is trusted.) Dataset-length / content hashing is explicitly deferred.

---

## 8. `DataProvider` changes (`datasets/data_provider.py`)

- Add attribute `self.split_weights = None` next to existing
  `self.split_indices = None` (line ~318).
- **Remove** `self.split_fraction` and stop reading `split_fraction` from the
  request dict in `prepare_datasets` (lines ~552). Keep
  `self.primary_data_location` (still needed by the driver to group by source).
- Update `__repr__` to surface split/balance info from the attached arrays, e.g.
  when `split_indices is not None`: print `len(split_indices)` selected items and,
  when `split_weights is not None`, that the group is rebalanced (e.g. number of
  classes / min–max weight). Drop the old `split_fraction` print.

---

## 9. `setup_dataset` changes (`pytorch_ignite.py`)

Simplify to **only** build providers:

```python
def setup_dataset(config, *, splits=None) -> dict[str, DataProvider]:
    data_request = generate_data_request_from_config(config)
    keys = splits if splits is not None else tuple(data_request.keys())
    return {k: DataProvider(config, data_request[k]) for k in keys if k in data_request}
```

- **Remove** the `providers_by_location` block (lines ~97–113) and the
  `shuffle` parameter (split computation — and thus shuffling — now lives in the
  driver / `dist_data_loader`). Update infer/test callsites that pass
  `shuffle=False` accordingly.
- **Move** `create_splits_from_fractions` out of `pytorch_ignite.py`; its
  non-stratified logic becomes the "Non-stratified" branch of
  `splitting_utils._compute_splits` (§7.3). Delete it from `pytorch_ignite.py`.

---

## 10. New verb: `CreateSplits` (`src/hyrax/verbs/create_splits.py`)

```python
@hyrax_verb
class CreateSplits(Verb):
    cli_name = "create_splits"
    description = "Compute (and persist) reproducible dataset splits and balance weights."
    REQUIRED_DATA_GROUPS = ()          # any groups present are processed
    OPTIONAL_DATA_GROUPS = ()

    @staticmethod
    def setup_parser(parser): ...      # no CLI opts initially

    def run_cli(self, args=None):
        print(self.run())

    def run(self) -> dict[str, DataProvider]:
        from hyrax.config_utils import create_results_dir, log_runtime_config
        from hyrax.pytorch_ignite import setup_dataset
        from hyrax.splitting_utils import create_splits
        datasets = setup_dataset(self.config)                # all groups
        results_dir = create_results_dir(self.config, "splits")
        create_splits(self.config, datasets, results_dir=results_dir, persist=True)
        log_runtime_config(self.config, results_dir)
        return datasets                                      # like h.prepare()
```

- Registered in `VERB_REGISTRY`; exposed automatically as `h.create_splits()` via
  `Hyrax.__getattr__` (`hyrax.py:268`).
- Returns `{group: DataProvider}` with `split_indices`/`split_weights` attached —
  same shape as `h.prepare()`.
- This satisfies CLAUDE.md "only add a verb if specifically told to" — the design
  explicitly mandates `create_splits`.

---

## 11. `dist_data_loader` changes (`pytorch_ignite.py`)

PR #944 already removed the `split` parameter and the legacy multi-split path;
the post-#944 signature is `dist_data_loader(dataset, config, shuffle=False)`
with a single code path. This spec reworks that body to use `Subset` + a
weight-aware sampler, and **drops the `indexes` element from the return** —
the function returns just the `DataLoader` (§3 D8).

```python
from torch.utils.data import Subset, WeightedRandomSampler

def make_sampler(n: int, weights, sampler_shuffle: bool):
    # n == len(subset); sampler indexes the SUBSET (local 0..n-1).
    if weights is not None:                      # balancing active
        return WeightedRandomSampler(
            weights=list(weights), num_samples=n,
            generator=torch_rng, replacement=True,
        )
    if sampler_shuffle:                          # uniform shuffle, no replacement
        return SubsetRandomSampler(range(n), generator=torch_rng)
    return None                                  # sequential order over the Subset

indexes = list(range(len(dataset)))
weights = None
if isinstance(dataset, DataProvider) and dataset.split_indices is not None:
    indexes = dataset.split_indices
    weights = dataset.split_weights              # ndarray or None
sub_dataset = Subset(dataset, indexes)
sampler = make_sampler(len(indexes), weights, shuffle)
return idist.auto_dataloader(sub_dataset, sampler=sampler, **data_loader_kwargs)
```

Key points (see §3 D6/D7/D8):

- **Subset-local indexing.** The sampler now indexes into `Subset`, so it yields
  `0..len(indexes)-1`; `Subset` maps those to the real dataset indices. This is
  what lets WRS carry a weight array sized to the *subset*, not the full dataset
  — the memory win the design wants.
- **No-replacement uniform default preserved.** When `weights is None` and
  `shuffle=True`, we use `SubsetRandomSampler(range(n))` (a per-epoch
  permutation, no replacement) — **byte-for-byte the current default training
  behavior**. WRS (`replacement=True`) is used **only** when balancing weights
  exist, where oversampling-with-replacement is the intended effect.
- **Sequential** (inference/test, `shuffle=False`, no weights): `sampler=None`;
  `Subset` preserves `indexes` order (equivalent to today's
  `SubsetSequentialSampler(indexes)`).
- `object_id` and collate flow are unaffected: `Subset.__getitem__` delegates to
  `DataProvider.__getitem__`, and `collate_fn` remains `dataset.collate`.
- **Return value is the bare `DataLoader`.** No caller uses the old second
  element (every callsite unpacks `loader, _ = …`), and the indexes/weights are
  available on the provider. Update all callsites to `loader = dist_data_loader(…)`:
  `verbs/train.py`, `verbs/test.py:105`, `verbs/infer.py:92`,
  `verbs/to_onnx.py:101`.
- The legacy `str`/`list[str]` split path and `[data_set]`-size handling are
  already gone (PR #944); nothing to preserve here.

---

## 12. Verb integration

Each verb that consumes splits follows the same pattern: `setup_dataset` →
`splitting_utils.create_splits` (attaches `split_indices`/`split_weights`) →
`dist_data_loader(provider, config, shuffle=…)`.

- **`train`** (`verbs/train.py`):
  - After `setup_dataset(...)`, call
    `create_splits(config, dataset, results_dir=results_dir, persist=True)` so the
    split files + `split_config.toml` land in the **train results directory**.
    When an equivalent split is reused from another results dir (§7.1 step 3),
    **copy** its `.npz` files and `split_config.toml` into the train results
    dir — decided in review: a training run's directory must be a complete,
    self-contained record of the splits used.
  - Post-#944, train has a single loader loop over its data groups; it becomes
    `dist_data_loader(dataset[group], config,
    shuffle=(group == "train" and train_shuffle))`, storing the bare loader.
- **`infer`** (`verbs/infer.py`): `setup_dataset(splits=("infer",))` →
  `create_splits(...)` (infer branch ⇒ first-N, no shuffle, weights None) →
  `dist_data_loader(dataset, config)` (sequential). Persisting is optional
  for infer; default `persist=False` unless we want a record (§17).
- **`test`** (`verbs/test.py`): like infer but group `test`; sequential loader.
- **`engine`** (`verbs/engine.py`): already reads `infer_dataset.split_indices`
  (line ~131); switch its split computation to the driver and keep reading
  `split_indices`.
- **`to_onnx`** (`verbs/to_onnx.py:101`): unpack change only.

All callsites change from `loader, _ = dist_data_loader(...)` to
`loader = dist_data_loader(...)` (§3 D8).

---

## 13. Non-training actions, warnings

- **`infer`**: defining any augmentation is a **hard error** (augmentation doc);
  `create_splits` for `infer` never rebalances and never shuffles.
- **`validate`/`test`**: if resampling is active for these groups (i.e. they are
  in `groups_to_balance`, §4.2), **warn** that
  validation/test metrics will diverge from real inference performance, but
  proceed. (Test-Time Augmentation is a legitimate reason to allow it.)

---

## 14. Backward compatibility & deprecations

- Untouched `[split]`/`[balance]` ⇒ all fractions `1.0`, no balance ⇒ identical
  to today (each group uses its full primary dataset; uniform no-replacement
  shuffle for train).
- Old configs with `data_request.*.split_fraction` are auto-migrated to `[split]`
  (migration 003) with the standard deprecation log.
- Legacy `[data_set].{train,validate,test}_size` + `pytorch_ignite.create_splits`
  are **removed in PR #944** (the legacy keys hard-error in `setup_dataset`).
  This spec builds on the post-#944 codebase and adds nothing for them.
- `[train].split` / `[infer].split` legacy string keys are untouched and out of
  scope (candidate for later removal; the default config already flags the infer
  one).

---

## 15. Testing plan

New `tests/hyrax/test_splitting_utils.py` (fast, using `HyraxRandomDataset` whose
`provided_labels = [0, 1, 2]` gives a class field):

1. **Defaults reproduce current behavior** — no `[split]`/`[balance]`: every
   group gets all indices; weights `None`; train loader yields a permutation
   (no duplicates) over an epoch.
2. **Custom fractions** — `split = {train:0.8, validate:0.2}` on a shared
   `data_location`: non-overlapping, correctly sized, deterministic for a fixed
   `rng_seed`; `Σ < 1.0` leaves a subset unused.
3. **Σ fractions > 1.0** on shared location ⇒ `RuntimeError`.
4. **Stratified** (`balance.field="label"`, no `groups`): each split preserves
   class ratios; weights `None`.
5. **Equal rebalance** (`groups=["train"]`, empty distribution): train
   `split_weights` ∝ inverse class frequency; validate weights `None`.
6. **Custom distribution** (`{label_0:0.25, label_1:0.75}`): weights match the
   raw `target_c / count_c`; with `groups=[]`, the distribution applies to
   **all** non-infer groups (§4.2 table); sum-to-1.0 violation ⇒ raise
   (pre-scan); unknown label ⇒ raise (post-scan, §7.2); missing label ⇒ warn +
   weight 0.
7. **Paths input** — generate a split dir, then point `split.*` at the `.npz`
   files: loads without recompute; mixed float+path ⇒ raise; differing parent
   dirs ⇒ raise.
8. **Equivalency reuse** — running twice with identical config reuses the first
   split dir (no second full scan); changing `rng_seed`/`field`/`distribution`/a
   group's `dataset_class`/`data_location`/fraction/`balance.groups` membership
   ⇒ not equivalent.
9. **`dist_data_loader`** — `Subset` length == `len(indexes)`; with weights,
   sampler is `WeightedRandomSampler(replacement=True)` and over-samples the
   minority class in expectation; without weights + shuffle, no duplicates.
10. **Persisted artifacts** — `<group>_split.npz` (always `indexes`; `weights`
    present only for balanced groups, absent ⇒ `None` on load) and
    `split_config.toml` (data_request/split/balance) exist and round-trip.

Plus `tests/hyrax/test_config_migrations.py`: migration 003 legacy-trigger and
clean-noop cases (§6.1).

Run: `python -m pytest -m "not slow"`, then `ruff check/format` and
`pre-commit run --all-files`.

---

## 16. Files created / modified

**Created**
- `src/hyrax/splitting_utils.py` — driver, validation, split/stratify, weights,
  persistence, equivalency.
- `src/hyrax/verbs/create_splits.py` — `CreateSplits` verb.
- `src/hyrax/config_migrations/migrations/003_move_split_fraction_to_split.py`.
- `tests/hyrax/test_splitting_utils.py`.

**Modified**
- `src/hyrax/hyrax_default_config.toml` — add `[split]` and `[balance]`
  (+`[balance.distribution]`) tables with defaults.
- `src/hyrax/pytorch_ignite.py` — simplify `setup_dataset`; remove
  `create_splits_from_fractions`; rework `dist_data_loader` (`Subset` + WRS,
  return the bare loader).
- `src/hyrax/datasets/data_provider.py` — add `split_weights`; remove
  `split_fraction`; update `__repr__`.
- `src/hyrax/config_schemas/data_request.py` — drop `split_fraction` field,
  its validator, and `validate_cross_group` split logic.
- `src/hyrax/verbs/{train,infer,test,engine}.py` — call the split driver; update
  loader wiring; train persists/copies split files to its results dir.
- `src/hyrax/verbs/to_onnx.py` — loader unpack change only (§3 D8).
- `src/hyrax/verbs/verb_registry.py` — `validate_data_request` no longer calls
  the removed `validate_cross_group` split path.
- `src/hyrax/config_utils.py` — `DYNAMIC_KEY_TABLES` (optional warning
  suppression).
- `tests/hyrax/test_config_migrations.py` — migration 003 tests.
- `docs/dataset_splits.rst` — document `[split]`/`[balance]` and `create_splits`.

---

## 17. Resolved decisions & open questions

Resolved during review (now normative in the body of this spec):

1. **Train results copy vs. write-in-place** — **copy**. When `train` reuses an
   equivalent pre-existing split dir, copy the `.npz` files + `split_config.toml`
   into the train results dir for a self-contained record (§12).
2. **Weight normalization** — store the **raw** `target_c / count_c` values; no
   normalization (§5).
3. **`.npz` "no weights" sentinel** — **omit the `weights` array** entirely for
   unbalanced groups; `load_split_files` treats its absence as `None` (§7.4).
4. **Legacy splitting code** — removed separately in **PR #944**; this spec
   assumes the post-#944 codebase (§2, §11, §14).

Still open:

1. **Infer persistence.** Should `create_splits` persist for `infer`/`test` by
   default, or only for `train`/explicit `h.create_splits()`? This spec defaults
   `persist=False` for infer/test.
