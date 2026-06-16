# Implementation Plan: Remove Legacy Split Configurations

This plan implements the spec in `specs/remove_legacy_splits.md`. The goal is to remove
the deprecated `data_set.train_size`, `data_set.validate_size`, and `data_set.test_size`
configuration keys and replace them with a fatal error guiding users to `split_fraction`
in `data_request`.

## Scope Summary

**12 files** reference the legacy split keys. After this work:
- 3 source files are modified (pytorch_ignite.py, train.py, hyrax_default_config.toml)
- 1 test file is deleted (test_splits.py)
- 2 test files are modified (test_train.py, test_split_fraction_integration.py)
- 1 doc file is rewritten (dataset_splits.rst)
- 3 archived notebooks have text updated (markdown cells only)

---

## Phase 1: Remove Legacy Keys from Default Config AND Add Fatal Error

These two steps MUST happen together. The runtime config is formed by merging the
default config with the user config. If we add the fatal-error check (1B) while the
default config still defines `train_size = 0.6` (etc.), the check will fire on
every run — even when the user never set those keys. Removing the keys from
defaults (1A) and adding the check (1B) must be done in the same pass.

### 1A. `src/hyrax/hyrax_default_config.toml` — remove legacy split keys

In the `[data_set]` section (around lines 269-290), delete everything related to the
three legacy keys. Specifically delete these lines and their preceding comment blocks:

```toml
# train_size, validation_size, and test_size use these conventions:
# * A `float` between `0.0` and `1.0` is the proportion of the dataset to include in the split.
# * An `int`, represents the absolute number of samples in the particular split.
# * It is an error for these values to add to more than 1.0 as ratios or the size
#   of the dataset if expressed as integers.

# Size of the train split. If `false`, the value is automatically set to the
# complement of test_size plus validate_size (if any).
train_size = 0.6

# Size of the validation split. If `false`, and both train_size and test_size
# are defined, the value is set to the complement of the other two sizes summed.
# If `false`, and only one of the other sizes is defined, no validate split is created.
validate_size = 0.2

# Size of the test split. If `false`, the value is set to the complement of train_size plus
# the validate_size (if any). If `false` and `train_size = false`, test_size is set to `0.25`.
test_size = 0.2
```

**Keep `seed = false`** — it is still used by `create_splits_from_fractions()` and is
not part of the legacy split system.

### 1B. `src/hyrax/pytorch_ignite.py` — add fatal error check in `setup_dataset()`

Add a check as the **very first thing** inside `setup_dataset()`, before the
`data_request = generate_data_request_from_config(config)` call on line 86.

Place `_LEGACY_SPLIT_KEYS` as a **module-level constant** near the top of the file
(e.g., after the `logger` definition on line 28):

```python
_LEGACY_SPLIT_KEYS = ("train_size", "validate_size", "test_size")
```

Then at the top of `setup_dataset()`, before any other logic:

```python
found = [k for k in _LEGACY_SPLIT_KEYS if k in config.get("data_set", {})]
if found:
    raise RuntimeError(
        f"Legacy split configuration keys found in [data_set]: {found}\n\n"
        "The train_size/validate_size/test_size configuration style has been removed.\n"
        "Please migrate to split_fraction in your [data_request] groups.\n\n"
        "Example:\n"
        "  [data_request.train.data]\n"
        "  dataset_class = 'YourDataset'\n"
        "  data_location = '/path/to/data'\n"
        "  primary_id_field = 'id'\n"
        "  split_fraction = 0.6\n\n"
        "  [data_request.validate.data]\n"
        "  dataset_class = 'YourDataset'\n"
        "  data_location = '/path/to/data'\n"
        "  primary_id_field = 'id'\n"
        "  split_fraction = 0.2\n\n"
        "For more information, see: https://hyrax.readthedocs.io/"
    )
```

This catches both file-based configs (user TOML with these keys) and programmatic
usage (e.g., `h.config["data_set"]["train_size"] = 0.6`). Because the keys have been
removed from the default config (1A), they will only appear in the runtime config if
the user explicitly provides them.

---

## Phase 2: Remove Legacy Code from `pytorch_ignite.py`

All changes in this phase are in `src/hyrax/pytorch_ignite.py`.

### 2A. Delete the `create_splits()` function (lines 272-403)

Delete the entire `create_splits()` function. This is the function that implements
the legacy percentage-based splitting logic. It is only called from the legacy path
in `dist_data_loader()`.

**DO NOT delete `create_splits_from_fractions()`** (lines 406-512). That function
implements the *current* split_fraction approach and must stay. The two functions
have similar names — only `create_splits` (without `_from_fractions`) is removed.

### 2B. Simplify `dist_data_loader()` (lines 157-269)

The current function has two code paths controlled by the `split` parameter:
- `split=False` (line 222): the current path using DataProvider + split_indices
- `split=str/list` (lines 244-269): the legacy path using `create_splits()`

After this change, only the `split=False` path remains. Here is exactly what to do:

1. **Remove the `split` parameter** from the function signature. Change:
   ```python
   def dist_data_loader(dataset, config, split=False, shuffle=False):
   ```
   to:
   ```python
   def dist_data_loader(dataset, config, shuffle=False):
   ```

2. **Remove the `Union` import** from the top of the file (line 6) if it becomes
   unused after this change. Check whether anything else in the file uses `Union`.

3. **Delete the legacy code path entirely** (lines 237-269): everything from the
   `# NOTE: The logic below is deprecated` comment through the end of the function.
   This includes the `isinstance(split, str)` check, the `create_splits()` call,
   the multi-split sampler/dataloader creation, and the dict return logic.

4. **Remove the `if isinstance(split, bool):` guard** (line 222). After removing the
   `split` parameter, the code that was inside this `if` block becomes the
   unconditional body of the function. Do NOT leave a vestigial `if True:` or
   `if isinstance(split, bool):` — just promote the block body up one indentation
   level.

5. **The function now always returns `(dataloader, indices)` tuple.** It no longer
   ever returns a dict. Update the docstring to reflect this simpler return type.

After these changes, the body of `dist_data_loader()` should be approximately:

```python
def dist_data_loader(dataset, config, shuffle=False):
    """Create Pytorch Ignite distributed data loaders.
    ...
    Returns
    -------
    tuple[DataLoader, list[int]]
        The distributed dataloader and the list of indices it samples from.
    """
    data_loader_kwargs = dict(config["data_loader"])
    if "shuffle" in data_loader_kwargs:
        # ... existing warning and pop logic ...

    data_loader_kwargs["collate_fn"] = dataset.collate

    torch_rng = torch.Generator(device=idist.device())
    seed = config["data_set"]["seed"] if config["data_set"]["seed"] else None
    if seed is not None:
        torch_rng.manual_seed(seed)

    def make_sampler(indexes, sampler_shuffle):
        if not indexes:
            return None
        if sampler_shuffle:
            return SubsetRandomSampler(indexes, generator=torch_rng)
        return SubsetSequentialSampler(indexes)

    indexes = list(range(len(dataset)))
    if isinstance(dataset, DataProvider) and dataset.split_indices is not None:
        indexes = dataset.split_indices

    sampler = make_sampler(indexes, shuffle)
    return idist.auto_dataloader(dataset, sampler=sampler, **data_loader_kwargs), indexes
```

### 2C. Update all callers of `dist_data_loader()`

After removing the `split` parameter, every call site that was passing a value for
`split` (typically `False`) must be updated. The callers are:

**`src/hyrax/verbs/train.py`** — updated in Phase 3 below (the call sites are
being rewritten as part of the train verb simplification).

**`src/hyrax/verbs/infer.py` line 92** — change:
```python
data_loader, _ = dist_data_loader(dataset, config, False)
```
to:
```python
data_loader, _ = dist_data_loader(dataset, config)
```

**Test files** — search for `dist_data_loader(` in all test files and remove the
`split` argument. The specific test files that call `dist_data_loader` are:
- `tests/hyrax/test_train.py` (being rewritten in Phase 4)
- `tests/hyrax/test_split_fraction_integration.py` (multiple call sites — updated
  in Phase 4)

---

## Phase 3: Simplify `src/hyrax/verbs/train.py`

The train verb currently has three code paths for creating dataloaders (lines 112-197):
- Path 1: Separate dataset groups without split_fraction
- Path 2: split_fraction on shared data (split_indices set by setup_dataset)
- Path 3: Legacy percentage-based splits (calls dist_data_loader with split=list)

After removing Path 3, Paths 1 and 2 have **identical dataloader creation logic** —
both iterate over dataset_splits and call `dist_data_loader(dataset[name], config,
shuffle=...)` per group. The only difference is whether `split_indices` is set on
the DataProvider, and `dist_data_loader` handles that transparently. So all three
paths collapse into one loop.

### What to remove

Delete the following (lines 112-197 approximately):

1. The large comment block explaining three paths (lines 112-131)
2. The `all_splits` variable (line 140) — it was only needed for the legacy path
   which passed it to `dist_data_loader(dataset["train"], config, all_splits, ...)`.
   After removal, only `dataset_splits` is needed.
3. The `has_split_groups` detection (lines 146-149)
4. The `if has_split_groups:` / `elif len(dataset) > 1:` / `else:` branching
   (lines 153-197)
5. The `import warnings` at the top of the file (line 2) — it was only used for the
   deprecation warning in Path 3. Verify nothing else in the file uses `warnings`
   before removing.

### What to replace it with

Replace lines 109-197 (from `train_shuffle = ...` through the end of the dataloader
creation) with:

```python
train_shuffle = config["train"]["shuffle"]

dataset_splits = [
    s for s in Train.REQUIRED_DATA_GROUPS + Train.OPTIONAL_DATA_GROUPS if s in dataset
]

data_loaders: dict[str, tuple] = {}
for split_name in dataset_splits:
    data_loaders[split_name] = dist_data_loader(
        dataset[split_name],
        config,
        shuffle=split_name == "train" and train_shuffle,
    )
```

Note: `dist_data_loader` no longer takes a `split` argument (Phase 2B removed it).

---

## Phase 4: Update Tests

### 4A. Delete `tests/hyrax/test_splits.py`

This file contains 12 tests that exclusively test `create_splits()`. Since that
function is being deleted, remove the entire file.

### 4B. Rewrite `test_train_percent_split()` in `tests/hyrax/test_train.py`

The existing test (lines 147-216) exercises the legacy split path end-to-end.
**Rename** it to `test_train_legacy_split_keys_raise_error` and **rewrite** it to
verify the fatal error behavior.

The rewritten test should be much simpler than the original — it only needs to:
1. Create a minimal config (via `hyrax.Hyrax()` or by building a config dict)
2. Set `config["data_set"]["train_size"] = 0.6` (one key is enough to trigger)
3. Call `setup_dataset(config)` inside `pytest.raises(RuntimeError, match=...)`
4. Assert the error message mentions "train_size" and "split_fraction"

Do NOT keep the full training setup, data_request configuration, or `h.train()` call
from the original test. The error fires at the top of `setup_dataset()` before any
dataset loading happens, so minimal config is sufficient.

Example:

```python
def test_train_legacy_split_keys_raise_error(tmp_path):
    """Setting legacy split keys in [data_set] raises a RuntimeError
    directing the user to split_fraction in [data_request]."""
    import hyrax
    from hyrax.pytorch_ignite import setup_dataset

    h = hyrax.Hyrax()
    h.config["data_set"]["train_size"] = 0.6

    with pytest.raises(RuntimeError, match="train_size"):
        setup_dataset(h.config)
```

This replaces both the old test_train_percent_split AND serves as the "new test for
fatal error at runtime" — there is no need for a separate test.

### 4C. `tests/hyrax/test_split_fraction_integration.py`

**Delete `test_legacy_multi_split_only_shuffles_train()`** (lines 559-584):
This test exercises the legacy multi-split code path in `dist_data_loader()` which
is being removed. Delete the entire test method.

**Keep `test_legacy_data_loader_shuffle_with_split_indices_does_not_error()`**
(lines 420-439): This test is about the `data_loader.shuffle` config warning, not
about the legacy split keys. It calls `dist_data_loader(dp, config, False)` — update
this call to `dist_data_loader(dp, config)` (remove the `False` argument since the
`split` parameter no longer exists).

**Update all other `dist_data_loader()` calls in this file**: Search for every call
to `dist_data_loader(` in the file and remove the `split` argument (the `False` that
was being passed as the third positional argument). The `shuffle` argument that some
calls pass should now become the second positional argument or be passed as a keyword.

For example, if a test has:
```python
dist_data_loader(dp, config, False, shuffle=True)
```
change to:
```python
dist_data_loader(dp, config, shuffle=True)
```

And if a test has:
```python
dist_data_loader(dp, config, False)
```
change to:
```python
dist_data_loader(dp, config)
```

---

## Phase 5: Update Documentation

### 5A. `docs/dataset_splits.rst` — full rewrite

The current content documents only the legacy split system. Replace it entirely with
documentation for the `split_fraction` approach. The new document should:

1. Keep the same RST reference label (`.. _dataset_splits:`) and title
2. Explain that splits are configured via `split_fraction` in `data_request` groups
3. Show a notebook example using `h.set_config("data_request", {...})` with
   `split_fraction` values (use the example from the spec in
   `specs/remove_legacy_splits.md` lines 26-43)
4. Show a TOML config example with `[data_request.train.data]` and
   `[data_request.validate.data]` sections containing `split_fraction`
5. Document constraints: fractions sharing a `data_location` must sum to <= 1.0
6. Document the `seed` key in `[data_set]` for reproducible split randomization
7. Use the same RST conventions as the current file (tab-set for notebook/CLI examples)

### 5B. Archived notebooks (markdown cells only)

These three archived notebooks reference the legacy split keys in their **markdown
commentary cells** (not code cells). Update only the text to note that the legacy
approach has been removed. Do NOT modify code cells or re-execute the notebooks.

Since these are archived, add a brief note like: *"Note: The percentage-based split
configuration shown here (`train_size`, `validate_size`, `test_size`) has been removed.
Use `split_fraction` in `data_request` groups instead."*

Files and locations:
- `docs/archived_notebooks/pre_executed/model_input_3.ipynb` (line 10 in the raw
  JSON): The markdown cell discusses "Percentage-based dataset splits" using
  `data_set.train_size` etc.
- `docs/archived_notebooks/pre_executed/hsc_train_to_similarity_search.ipynb`
  (line 997): Text about setting `test_size` to 100% for inference.
- `docs/archived_notebooks/pre_executed/mpr_demo.ipynb` (line 1752): Same pattern
  as above.

---

## Verification

After all phases are complete, run the following checks:

1. **`ruff check src/ tests/ && ruff format src/ tests/`** — ensure no lint or
   formatting issues.
2. **`python -m pytest -m "not slow"`** — ensure all fast tests pass.
3. **`grep -r "train_size\|validate_size\|test_size" src/ tests/ docs/`** — verify
   no remaining references except:
   - The fatal error message text in `src/hyrax/pytorch_ignite.py`
   - `sklearn.train_test_split` in `docs/pre_executed/supervised_lightcurve_transients*.ipynb`
     (unrelated to hyrax config)
   - The archived notebook notes added in Phase 5B
4. **`pre-commit run --all-files`** — ensure pre-commit hooks pass.

---

## Out of Scope

- **`[infer].split` config key**: The default config has `split = "infer"` under
  `[infer]` with a TODO comment suggesting removal. This is a separate concern
  unrelated to the legacy `data_set.*_size` keys.
- **`data_set.seed` key**: Still actively used by `create_splits_from_fractions()`.
  Not part of this removal.
- **`supervised_lightcurve_transients.ipynb`** and `supervised_lightcurve_transients_gs.ipynb`:
  These reference `test_size` only in the context of `sklearn.train_test_split`,
  not hyrax configuration. No changes needed.
- **Config migrations**: No config migration is created for this change. The legacy
  split keys predate the migration system, and there is no clean automatic conversion
  path. Detection is handled at runtime in `setup_dataset()`.
