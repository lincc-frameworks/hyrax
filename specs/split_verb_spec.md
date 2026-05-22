# Specification: `split` verb — reproducible, stratified, weighted splits

## Context

Hyrax currently computes train/validate/test splits *implicitly* at run time. The modern
path (`create_splits_from_fractions` in `pytorch_ignite.py`) shuffles `range(len)` with the
configured seed and slices contiguous proportional segments; the legacy path uses
`data_set.train_size/...`. There is **no** stratification (class-balanced splits), **no**
`WeightedRandomSampler` / class weighting, and **no** way to persist a split or verify a
saved split still matches the current config. That makes class-imbalanced training awkward
and makes splits non-portable/non-auditable across runs.

This change adds a `split` verb that **precomputes** splits once, persists them as integer
index lists plus a compatibility fingerprint and (for the train split) inverse-frequency
sample weights, and lets `train`/`test`/`infer` consume a premade split via a per-group
`data_request` field. When a verb loads a premade split it verifies compatibility with the
current config and fails clearly otherwise. Outcome: reproducible, shareable, optionally
stratified splits, with automatic `WeightedRandomSampler` for imbalanced training.

### Confirmed design decisions
1. Split files store **integer index lists** per split, a **fingerprint** (dataset length,
   class, location, seed, fractions, stratify field, probed object IDs), and — for the train
   split when `weights=true` — **inverse-frequency class weights**. The indices are
   **relative to the primary dataset** of the data group that consumes the split. (Optimization:
   weights can store class counts instead of precomputed inverse frequencies to save space;
   runtime converts counts → inverse frequencies as needed.)
2. A premade split is referenced by **extending `data_request` per group** with a single new
   `split_file` field. The split verb writes **one self-describing file per split**, so each
   group points only at the file it consumes (no `split_name` needed). Each split file's indices
   and weights are computed and valid **for that group's primary dataset**.
3. The stratify label is found via a **config key naming a getter** (`stratify_field = "label"`
   → the dataset's `get_label`).
4. The split verb computes inverse-frequency train weights; `dist_data_loader` **auto-selects
   `WeightedRandomSampler`** when weights are present, else the existing samplers.

## Implementation approach

Three layers: a reusable pure-function module, a thin verb, and small wiring changes in the
data_request schema + dataloader.

### 1. New module `src/hyrax/splits.py`
Pure functions (numpy/json + `DataProvider` duck-typing; kept out of `pytorch_ignite.py` to
avoid import cycles and give a clean test surface):

- `compute_split_indices(n, fractions, *, seed=None, labels=None) -> dict[str, list[int]]`
  - `labels is None`: replicate `create_splits_from_fractions` exactly (`np.random.seed(seed)`,
    shuffle `range(n)`, contiguous proportional slices in `fractions` insertion order, leftover
    to last split when total≈1.0) so a random premade split is byte-identical to the implicit path.
  - `labels` given: stratify — bucket indices by label, shuffle+slice each bucket by the same
    fractions with the seeded RNG, then re-shuffle each split so classes interleave.
- `compute_class_counts(labels, train_indices) -> list[int]` — per-sample class counts
  **positionally aligned to `train_indices`** (count class frequency over train only; each sample
  gets the count of its class in the train split). Returns `int`s to save space; runtime converts
  to inverse frequencies as `weight = 1.0 / count` when loading the split. Be clear in the 
  docstring that the returne value needs to be inverted to be used with WeightedRandomSampler.
- `compute_split_fingerprint(*, dataset_len, dataset_class, data_location, seed, fractions,
  stratify_field, id_getter, n_probes=8) -> str` — sha256, modeled on `_join_cache_fingerprint`
  (`data_provider.py:111`): version + scalars + sorted fractions + spread-out object-ID probes.
- `save_split(output_dir, split_indices, train_class_counts, meta) -> dict[str, Path]` — writes
  **one `<split>.npz` per split** (e.g. `train.npz`, `validate.npz`, `test.npz`) into
  `output_dir`, each self-describing; returns `{split_name: path}`.
- `load_split(path) -> (indices, class_counts, meta)` — loads a single split file.
- `verify_split_compatible(meta, provider, config, *, group_name) -> None` — recompute
  fingerprint and compare; on mismatch raise `RuntimeError` naming the differing scalar field(s);
  also range-check indices against `len(provider)`.

**Artifact format:** one `.npz` per split, written into `create_results_dir(config, "split")`
alongside `runtime_config.toml`. Each file holds `indices` (int64), `meta_json` (0-d string),
and — for `train.npz` only — `class_counts` (int32). Per-split files match the user's "files
containing index lists" model: a consuming group points `split_file` directly at the file it
needs, no name selection. Each file's `meta` carries the **same shared fingerprint** (so
`verify_split_compatible` works per file) plus a `split_set_id` identifying the generation, so a
mismatched mix (e.g. a `train.npz` from one generation with a `validate.npz` from another) can be
detected. A readable `split_meta.json` summarizing the whole set is also written for diagnostics.

### 2. New verb `src/hyrax/verbs/split.py`
`@hyrax_verb class Split(Verb)`, `cli_name="split"`, `REQUIRED/OPTIONAL_DATA_GROUPS = ()`. The
verb derives its working set dynamically (the data_request groups that declare `split_fraction`)
and validates them itself, so it does not declare static required groups (which would force
specific names and run the base partition logic prematurely). `setup_parser` = `pass`; `run_cli`
calls `run()` and prints the artifact path. `run()`:
1. Read `[split]` config: `stratify_field` (`false`→None) and `weights` (bool). **Fractions and
   group names are not in `[split]`** — they come from `data_request`.
2. Build the partition spec from `data_request`: `fractions = {group: cfg["split_fraction"] for
   group, cfg in data_request groups if split_fraction is set}` (e.g. `{"train":0.6,
   "validate":0.2,"test":0.2}`). Error if no group declares `split_fraction`.
3. `results_dir = create_results_dir(config,"split")`; `log_runtime_config(config, results_dir)`.
4. `providers = setup_dataset(config, splits=tuple(fractions))`; for the groups declaring
   `split_fraction`, verify they share the same `primary_data_location` (all groups must consume
   the same underlying primary dataset, or splits are incompatible). Take one provider as the
   source. `n = len(provider)` — the full length of that **primary dataset** (note:
   `__len__` returns the primary dataset size regardless of any `split_indices` the partition
   loop set, verified in `data_provider.py:383`); `seed = config["data_set"]["seed"] or None`.
   The computed split indices will index into this primary dataset.
5. If `stratify_field`: resolve `provider.dataset_getters[provider.primary_dataset][stratify_field]`
   (`data_provider.py:519`); read `labels=[getter(i) for i in range(n)]`. **Fail loud** if the
   getter is missing (list available fields) or returns `None` for any item (do **not** silently
   fall back to random). Validate that all class counts are ≥ the number of splits; if any class
   has fewer instances than the number of splits, log a **warning** (stratification may not be
   possible for that class across all splits). Else `labels=None`.
6. `split_indices = compute_split_indices(n, fractions, seed=seed, labels=labels)`.
7. `class_counts = compute_class_counts(labels, split_indices["train"])` when
   `weights and labels is not None and "train" in split_indices`, else `None` (class counts apply to
   the conventionally-named `train` split; log if requested but no labels / no train group).
   At runtime, when loading a split with counts, they are converted to inverse frequencies as
   `weight = 1.0 / count` for use with `WeightedRandomSampler`.
8. Build `meta` (fingerprint via `provider.get_object_id` as `id_getter`, plus a `split_set_id`)
   → `save_split(results_dir, split_indices, train_class_counts, meta)` → return `results_dir` (the
   per-split files live inside it).

**Registration:** decorator auto-registers; also add `from hyrax.verbs.split import Split` and
`"Split"` to `src/hyrax/verbs/__init__.py`. CLI (`hyrax_cli/main.py`) and notebook API
(`Hyrax.split` via `hyrax.py:268`) auto-discover.

### 3. `data_request.py` schema (`DataRequestConfig`)
`data_request` is the only pydantic-validated config key (`config_utils.py:182`), so adding an
optional field here is correct. Add exactly one field: `split_file: str|None` (with a
`@field_validator` that resolves the path like `resolve_data_location`, local-only). The indices
in a `split_file` are **relative to the primary dataset** named in that group's `data_request`.
Add `@model_validator`s: `split_file` XOR `split_fraction` (mutually exclusive); `split_file`
requires `primary_id_field`. Update `validate_cross_group` (`data_request.py:269`) so the
"all-or-none split source" consistency check treats `split_file` like `split_fraction` (groups
consistently using `split_file` pass). Note: `stratify_field` is **not** a data_request field —
it is only used by the split verb at generation time and read from the `[split]` config section
(below); consuming verbs need only `split_file`.

### 4. `pytorch_ignite.py` wiring
- **`setup_dataset`** (`:53`): after providers are built (~`:95`) and before the `split_fraction`
  partition loop, add a pass: for each group whose primary config has `split_file`, call
  `(indices, class_counts, meta) = load_split(Path(split_file))` then
  `verify_split_compatible(meta, provider, config, group_name=...)`, and set
  `provider.split_indices = indices` and `provider.class_counts = class_counts` (only `train.npz`
  carries counts; others load `None`). Premade-split groups have no `split_fraction`, so the
  existing partition loop (`:104`) already skips them.
- **`create_splits_from_fractions`** (refactor, `:406`): replace its implementation to use
  `compute_split_indices(n, fractions, seed=seed, labels=None)` internally. Extract
  `fractions = {name: provider.split_fraction for ...}` from the providers dict, validate inputs
  as before, then call the new function. This eliminates duplication (the split verb also uses
  `compute_split_indices` for random splits) while preserving the existing function signature and
  test compatibility.
- **`DataProvider.__init__`** (`data_provider.py:318`): add `self.class_counts = None` next to
  `self.split_indices = None`.
- **`dist_data_loader`** (`:222` boolean-split branch): when `dataset.split_indices` is set and
  `dataset.class_counts is not None and shuffle`, convert counts to inverse-frequency weights
  as `weights = [1.0 / c for c in class_counts]`, then wrap in `torch.utils.data.Subset(dataset,
  split_indices)` and use `WeightedRandomSampler(weights, num_samples=len(split_indices),
  replacement=True, generator=torch_rng)`, passing `collate_fn=dataset.collate` explicitly
  (Subset doesn't proxy `.collate`); return `indexes=split_indices`. Otherwise unchanged
  (`SubsetRandomSampler`/`SubsetSequentialSampler`). **Alignment note:** `WeightedRandomSampler`
  yields *positions*, so weights, the `Subset`, and `train_indices` must all be positionally
  aligned — guaranteed by `compute_class_counts`. Validate/test stay deterministic
  (class_counts only on train + branch requires `shuffle=True`, which Train passes only for train).
  Train needs **no** changes: its `has_split_groups` check (`train.py:146`) keys off
  `split_indices`, so premade splits flow through the existing path-2 loop.

### 5. `hyrax_default_config.toml`
Add a plain (non-pydantic) `[split]` section near `[data_set]` (`:249`). `false` = None. Only
`weights` and `stratify_field` live here — **fractions and group names come from the
`data_request` groups**, so `[split]` has no `group` key and no `[split.fractions]` table:
```toml
[split]
weights = true           # compute inverse-frequency train weights (needs stratify_field)
stratify_field = false   # getter field for class-balanced splits; false = random split
```
No migration needed (migrations are only for renaming/moving existing keys); new optional
data_request fields default to None and need no TOML default.

## Critical files to modify
- `src/hyrax/splits.py` (new) — split/weights/fingerprint/save/load/verify
- `src/hyrax/verbs/split.py` (new) — the verb; register in `src/hyrax/verbs/__init__.py`
- `src/hyrax/pytorch_ignite.py` — `setup_dataset` premade-split load (~`:95`); `dist_data_loader` weighted sampler (`:222`)
- `src/hyrax/config_schemas/data_request.py` — new fields + validators (`:53`, `:269`)
- `src/hyrax/datasets/data_provider.py` — `self.sample_weights = None` (`:318`)
- `src/hyrax/hyrax_default_config.toml` — `[split]` section

## Tests (`tests/hyrax/test_split_verb.py` — distinct from existing `test_splits.py`)
Unit: random proportions; determinism (seed); stratified proportions per split; class counts
(length/alignment/ratio); fingerprint sensitivity; save/load roundtrip (per-split files;
only `train.npz` has counts); `verify_split_compatible` mismatch raises with named field;
missing/None label raises. Edge case: stratified split with class count < num_splits logs warning.
Integration: `Hyrax().split()` writes
`train.npz`/`validate.npz`/`test.npz`+`split_meta.json`+`runtime_config.toml`; split→train
end-to-end (HyraxRandomDataset with `provided_labels`) asserting the train sampler is
`WeightedRandomSampler` (with inverse-frequency weights computed from class counts) and validate/test are `SubsetSequentialSampler`. Add data_request
validation cases (`split_file`+`split_fraction` raises; `split_file` cross-group passes).

## Verification

```bash
python -m pytest -m "not slow" tests/hyrax/test_split_verb.py tests/hyrax/test_pytorch_ignite.py tests/hyrax/test_verb_data_request_validation.py
ruff check src/ tests/ && ruff format src/ tests/
python -m pytest -m "not slow"          # full fast suite (regression)
```

### Manual verification (notebook/CLI)
Configure a `HyraxRandomDataset` with `provided_labels` and
`data_request.{train,validate,test}` groups each carrying a `split_fraction` (the source of the
proportions), set `config["split"]["stratify_field"]="label"` and `config["split"]["weights"]=true`, 
run `h.split()` (returns a dir holding `train.npz`/`validate.npz`/`test.npz` with class counts in train); 
then switch each group from `split_fraction` to a `split_file` pointing at the matching file; run `h.train()` 
for 1 epoch and confirm it consumes the split, loads class counts, converts them to inverse-frequency weights, 
and uses `WeightedRandomSampler` for train. 

CLI example:
```bash
hyrax split -c cfg.toml && hyrax train -c cfg.toml
```

## Risks and mitigation

- **Legacy `split_fraction` vs premade split**: pydantic mutual-exclusion + the `split_fraction is
  not None` guard in `setup_dataset` keep paths disjoint; both converge on Train's path-2.
- **`WeightedRandomSampler` position-vs-index trap**: solved by `Subset` + positional alignment
  of class_counts (must pass `collate_fn=dataset.collate`). Counts are converted to inverse frequencies
  at load time to save space in the `.npz` file.
- **Non-stratifiable / None labels**: explicit `RuntimeError`, never silent random fallback.
- **Class count < number of splits**: logged as a **warning** during stratified split (e.g., if
  3 splits but only 2 instances of a class exist, not all splits can receive that class);
  stratification proceeds anyway, but the user is informed the result may be imbalanced for that class.
- **Fingerprint probing**: mirrors the trusted `_join_cache_fingerprint` tradeoff;
  `SPLIT_FORMAT_VERSION` invalidates old files on format change.
