# Plan: Splits and Dataset Balancing

## Context

Implementing `specs/splits_and_balancing_spec.md` on branch `awo/splits-and-balancing-spec`. Adds configuration-driven, reproducible dataset splitting and class rebalancing via two new TOML tables (`[split]`, `[balance]`), a new `splitting_utils.py` module, and a new `CreateSplits` verb.

**Key assumption:** PR #944 is treated as already merged. This means `dist_data_loader` already returns a bare `DataLoader` (no tuple), the `split` parameter is gone, the legacy `create_splits` function is removed, and `train.py` has no path-3 legacy code. The only legacy artifact remaining is `create_splits_from_fractions` in `pytorch_ignite.py`, which this spec moves to `splitting_utils.py`.

---

## Phase 1 — Config foundation

### 1a. `hyrax_default_config.toml`
Add new tables **after** the existing config sections:
```toml
[split]
# Group keys are absent by default; any group in data_request not listed
# here defaults to fraction 1.0 (full dataset).
rng_seed = false

[balance]
field  = ""
groups = []

[balance.distribution]

# [label]
# Optional. Maps human-readable string aliases to the raw values returned by
# get_<balance.field>.  Needed when get_<balance.field> returns non-string
# values (e.g. integers).  This table has no entries in the default config;
# users add their own alias = raw_value pairs.
```

### 1b. Config migration `003_move_split_fraction_to_split.py`
Create `src/hyrax/config_migrations/migrations/003_move_split_fraction_to_split.py`
- Decorator: `@migration_step(from_version=3)` (auto-derives `CURRENT_CONFIG_VERSION = 4`)
- Custom function body (not key_renames): iterates `data_request` groups, pops `split_fraction` from each primary dataset definition, writes to `cfg["split"][group_name]`
- Exact template is in spec §6.1

### 1c. `config_schemas/data_request.py`
- **Remove** `split_fraction` field (lines 33–38)
- **Remove** `require_primary_id_for_split_fraction` validator (lines 65–70)
- **Remove** `validate_cross_group` split_fraction checks (lines 256–288) — the sum and consistency checks move to `splitting_utils.validate_split_config`. Keep the method signature as a no-op or remove it; update `verb_registry.py` accordingly.

### 1d. `config_utils.py`
Add near `_validate_runtime_config` (line ~505):
```python
DYNAMIC_KEY_TABLES = ("split", "distribution", "label")
```
When recursing into a table whose name is in `DYNAMIC_KEY_TABLES`, skip the "no default" warning for its children. `"split"` covers non-standard group keys; `"distribution"` covers class-fraction keys; `"label"` covers user-defined alias keys.

---

## Phase 2 — `splitting_utils.py` (new file)

Create `src/hyrax/splitting_utils.py`. All functions listed below are public unless prefixed `_`.

### Validation helpers
```python
def validate_split_config(config, datasets) -> None
def validate_balance_config(config, datasets) -> None
def validate_distribution_labels(distribution, observed_labels) -> None
```
- `validate_split_config`: per-group float domain `(0,1]`; all-floats-or-all-paths; paths share parent dir; for groups sharing `primary_data_location`, `Σ ≤ 1.0`.
- `validate_balance_config`: pre-scan checks — `field` getter exists; `groups ⊆ data_request` (warn extras); `distribution` values in `(0,1]`, sum to 1.0. When `[label]` is non-empty also checks: (a) label values are unique (no two aliases map to the same raw value); (b) every key in `balance.distribution` appears in `[label]`.
- `validate_distribution_labels`: post-scan; unknown label in distribution → raise; observed label missing from non-empty distribution → warn + weight 0. Operates entirely in alias string space when `[label]` is present (re-keying has already been done in `_compute_splits`).

### RNG helper
```python
def _shuffle(indices, config) -> None:  # in-place
```
Reproduces exact current shuffle semantics when `rng_seed` is falsy (`false`, the default → global `np.random.seed` + `np.random.shuffle`). Uses `np.random.default_rng(rng_seed)` for an integer seed; a non-empty string raises `RuntimeError`.

### Core split computation
```python
def _compute_splits(config, datasets) -> dict[str, dict]
```
Groups datasets by `provider.primary_data_location`. For each location group:
- `infer`: first `round(N * fraction)` indices, no shuffle, weights None.
- Non-stratified (`balance.field` falsy): shuffle + contiguous slice (mirrors `create_splits_from_fractions` logic being moved here from `pytorch_ignite.py`).
- Stratified (`balance.field` set): build `class_inds` via `get_<field>` scan. If `[label]` is non-empty: build `raw_to_name = {v: k for k, v in config["label"].items()}` and re-key `class_inds` to alias strings; if a raw value is not covered by `[label]`, raise `RuntimeError`. Then call `validate_distribution_labels` (operates in alias space). Per-class shuffle + slice → compute weights per §5 for groups in `groups_to_balance`.

Compute `groups_to_balance` per spec §4.2 table:
```python
groups_to_balance = set(balance.groups) or (set(data_request) - {"infer"} if balance.distribution else set())
```

Returns `{group: {"indexes": ndarray[int64], "weights": ndarray[float64] | None}}`.

### Persistence / loading
```python
def persist_splits(results_dir, splits, config) -> None
def load_split_files(paths) -> dict[str, dict]
def assign_splits_to_providers(datasets, splits) -> None
```
- `persist_splits`: `np.savez_compressed` per group — always `indexes` (`int64`), `weights` (`float64`) only when not None. Also writes `split_config.toml` with `data_request`, `split`, `balance`, **and `label`** sections (so the persisted config is self-contained for equivalency checks and traceability).
- `load_split_files`: loads `.npz`; `weights = npz["weights"] if "weights" in npz.files else None`.
- `assign_splits_to_providers`: `provider.split_indices = indexes.tolist()`, `provider.split_weights = weights`.

### Equivalency
```python
def find_equivalent_split(config, results_root=None) -> dict[str, Path] | None
def configs_equivalent(prev, cur) -> tuple[bool, list[str]]
```
`find_equivalent_split` scans `results_dir/*-splits-*/split_config.toml` most-recent first, calls `configs_equivalent`, returns first match or None.

`configs_equivalent` compares (per spec §7.5): `balance.field`, `balance.distribution`, resolved `split.rng_seed`, and per-group `dataset_class`, `data_location`, `split.<group>`, `balance.groups` membership.

### Public driver
```python
def create_splits(config, datasets, *, results_dir=None, persist=True) -> dict[str, dict]
```
Driver flow (spec §7.1):
1. `validate_split_config` + `validate_balance_config`
2. If paths → `load_split_files`; compare sibling `split_config.toml` with `configs_equivalent`, log warning on differences; return
3. Else → `find_equivalent_split`; if found, load and return
4. Else → `_compute_splits`, `assign_splits_to_providers`, `persist_splits` if requested, return

---

## Phase 3 — `data_provider.py` changes

File: `src/hyrax/datasets/data_provider.py`

- **Add** `self.split_weights = None` next to `self.split_indices = None` (~line 318)
- **Remove** `self.split_fraction = None` (~line 312)
- **Remove** the `split_fraction` / `primary_data_location` assignment block in `prepare_datasets` (~lines 547–553). Keep `self.primary_data_location` — it is still needed by `splitting_utils` to group providers by source.
- **Update** `__repr__` (~lines 390–416): replace `split_fraction` printout with split info derived from attached arrays — show `len(split_indices)` selected items and, when `split_weights is not None`, note that the group is rebalanced.

---

## Phase 4 — `pytorch_ignite.py` changes

File: `src/hyrax/pytorch_ignite.py`

### `setup_dataset` simplification (~lines 53–113)
Replace with the minimal form from spec §9:
```python
def setup_dataset(config, *, splits=None) -> dict[str, DataProvider]:
    data_request = generate_data_request_from_config(config)
    keys = splits if splits is not None else tuple(data_request.keys())
    return {k: DataProvider(config, data_request[k]) for k in keys if k in data_request}
```
- Remove `shuffle` parameter and `providers_by_location` block.
- Remove `create_splits_from_fractions` (its logic moves to `splitting_utils._compute_splits`).

### `dist_data_loader` body rework (~lines 157–235, post-#944)
The post-#944 signature is already `dist_data_loader(dataset, config, shuffle=False)` returning a bare `DataLoader`. Update the body to use `Subset` + weight-aware sampler (spec §11):

```python
from torch.utils.data import Subset, WeightedRandomSampler

def make_sampler(n, weights, sampler_shuffle):
    if weights is not None:
        return WeightedRandomSampler(weights=list(weights), num_samples=n,
                                     generator=torch_rng, replacement=True)
    if sampler_shuffle:
        return SubsetRandomSampler(range(n), generator=torch_rng)
    return None

indexes = list(range(len(dataset)))
weights = None
if isinstance(dataset, DataProvider) and dataset.split_indices is not None:
    indexes = dataset.split_indices
    weights = dataset.split_weights
sub_dataset = Subset(dataset, indexes)
sampler = make_sampler(len(indexes), weights, shuffle)
return idist.auto_dataloader(sub_dataset, sampler=sampler, **data_loader_kwargs)
```

Key decisions from spec §3: WRS only when weights are present (D6); sampler indexes subset-local space (D7); bare DataLoader returned (D8).

---

## Phase 5 — New verb `CreateSplits`

Create `src/hyrax/verbs/create_splits.py` (spec §10):
```python
@hyrax_verb
class CreateSplits(Verb):
    cli_name = "create_splits"
    REQUIRED_DATA_GROUPS = ()
    OPTIONAL_DATA_GROUPS = ()
    ...
    def run(self):
        datasets = setup_dataset(self.config)
        results_dir = create_results_dir(self.config, "splits")
        create_splits(self.config, datasets, results_dir=results_dir, persist=True)
        log_runtime_config(self.config, results_dir)
        return datasets
```

---

## Phase 6 — Verb integration

All callsites already use bare `loader = dist_data_loader(...)` after PR #944.

### `verbs/train.py`
After `setup_dataset(...)`, call `create_splits(config, dataset, results_dir=results_dir, persist=True)`. When `create_splits` reuses an equivalent split from another dir, **copy** those `.npz` + `split_config.toml` into the train results dir (spec §12). Loader loop becomes:
```python
for group in dataset_splits:
    data_loaders[group] = dist_data_loader(dataset[group], config,
                                           shuffle=(group == "train" and train_shuffle))
```

### `verbs/infer.py` (~line 92)
`setup_dataset(splits=("infer",))` → `create_splits(config, dataset, persist=False)` → `dist_data_loader(dataset["infer"], config)` (sequential, no shuffle).

### `verbs/test.py` (~line 105)
Same pattern as infer, group `test`, `persist=False`.

### `verbs/engine.py` (~line 131)
Switch infer split computation to `create_splits`; keep reading `split_indices` from provider (no change to the read side).

### `verbs/to_onnx.py` (~line 101)
Loader unpack already handled by PR #944; no additional change needed unless the body still references `split`.

### `verbs/verb_registry.py`
Remove or no-op the `validate_cross_group` call that checked split_fraction sums — that validation now lives in `splitting_utils.validate_split_config`.

---

## Phase 7 — Tests

### `tests/hyrax/test_splitting_utils.py` (new file)
Ten tests per spec §15 using `HyraxRandomDataset` (has `get_label`, `provided_labels`):
1. Defaults reproduce current behavior (all indices, weights None, no-dup train permutation)
2. Custom fractions on shared `data_location` → non-overlapping, deterministic
3. Σ > 1.0 on shared location → RuntimeError
4. Stratified splits (`balance.field="label"`) → class ratios preserved, weights None
5. Equal rebalance (`groups=["train"]`, empty distribution) → train weights ∝ inverse frequency
6. Custom distribution + validation/checking; groups=[] with distribution → all non-infer groups; sum violation → raise; unknown label → raise; missing label → warn
6a. **`[label]` translation** — use a dataset whose `get_label` returns integers (e.g. `0`, `1`, `2`); define `[label]` mapping string aliases → those integers; set `balance.distribution` using alias strings. Verify: `class_inds` re-keyed to aliases; weights correct; `split_config.toml` round-trips `[label]`. Error cases: distribution key absent from `[label]` → raise (pre-scan); duplicate raw values in `[label]` → raise (pre-scan). Warning case: dataset raw value absent from `[label]` → log warning, items excluded, no error.
7. Paths input: load without recompute; mixed float+path → raise; differing parent dirs → raise
8. Equivalency reuse: identical config reuses split dir; any changed field → not equivalent
9. `dist_data_loader` Subset + sampler types (WRS when weights, SubsetRandomSampler when shuffle, None when sequential)
10. Persisted artifacts: `.npz` round-trip; `weights` absent when None; `split_config.toml` exists

### `tests/hyrax/test_config_migrations.py` (additions)
- Migration 003: v3 config with `split_fraction` → migrates to `[split]`, strips old key
- Migration 003: clean v4 config → no-op

---

## Critical files

| File | Action |
|------|--------|
| `src/hyrax/splitting_utils.py` | **Create** |
| `src/hyrax/verbs/create_splits.py` | **Create** |
| `src/hyrax/config_migrations/migrations/003_move_split_fraction_to_split.py` | **Create** |
| `tests/hyrax/test_splitting_utils.py` | **Create** |
| `src/hyrax/hyrax_default_config.toml` | Add `[split]`, `[balance]`, `[balance.distribution]` |
| `src/hyrax/pytorch_ignite.py` | Simplify `setup_dataset`; rework `dist_data_loader`; remove `create_splits_from_fractions` |
| `src/hyrax/datasets/data_provider.py` | Add `split_weights`; remove `split_fraction`; update `__repr__` |
| `src/hyrax/config_schemas/data_request.py` | Remove `split_fraction` field + validators |
| `src/hyrax/verbs/train.py` | Call `create_splits`; simplify loader loop; copy split files |
| `src/hyrax/verbs/infer.py` | Call `create_splits`; update loader |
| `src/hyrax/verbs/test.py` | Call `create_splits`; update loader |
| `src/hyrax/verbs/engine.py` | Switch to `create_splits` driver |
| `src/hyrax/verbs/to_onnx.py` | Verify no stale split ref |
| `src/hyrax/verbs/verb_registry.py` | Remove `validate_cross_group` split logic |
| `src/hyrax/config_utils.py` | Add `"split"` and `"label"` to `DYNAMIC_KEY_TABLES` |
| `tests/hyrax/test_config_migrations.py` | Add migration 003 tests |

---

## Verification

```bash
# Fast test suite
python -m pytest -m "not slow"

# Linting
ruff check src/ tests/ && ruff format src/ tests/

# Pre-commit
pre-commit run --all-files
```

Specific test markers to watch:
- `test_splitting_utils.py` — all 10 scenarios pass
- `test_config_migrations.py` — migration 003 trigger and no-op both pass
- Existing `test_pytorch_ignite.py` / `test_train.py` pass (regression check on default behavior)
