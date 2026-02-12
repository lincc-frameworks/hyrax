# Edge Cases for Split-Fraction Branch (`issue/663/define-split-fraction-in-data-request`)

PR #690: Updated `data_request` to support splits

---

## 🔴 High — Likely Bugs or Silent Data Corruption

### 1. Floating-point sum in `validate_split_fractions_within_location`
- **File:** `src/hyrax/config_schemas/data_request.py`
- Uses exact `> 1.0` comparison with no epsilon.
- Classic FP issue: `0.1 + 0.2 + 0.7 = 1.0000000000000002` → rejects a valid config.
- The runtime counterpart in `_get_split_indices` uses `min(end, total)` clamping, but the Pydantic validator will reject the config before it ever gets there.
- **Fix:** Use `math.isclose(total, 1.0)` or check `> 1.0 + epsilon`.

### 2. `_get_split_indices` takes `total` from only the first provider
- **File:** `src/hyrax/data_sets/data_provider.py`
- If providers for the same `data_location` have different lengths (e.g., different filter criteria or caching issues), split indices will silently point out-of-range or at wrong items in shorter providers.
- No assertion verifies uniform `len()` across providers.
- **Fix:** Assert all providers for the same `data_location` have the same length.

### 3. `shuffle=True` + `SubsetRandomSampler` conflict
- **File:** `src/hyrax/data_sets/dist_data_loader.py`
- When `split_fraction` triggers a `SubsetRandomSampler`, PyTorch's `DataLoader` raises if both `shuffle=True` and a sampler are provided.
- The `train` and `infer` verbs guard against this, but `dist_data_loader` itself does **not** guard on the code paths where it passes `shuffle` through.
- **Fix:** In `dist_data_loader`, force `shuffle=False` when a sampler is present.

### 4. ONNX inference ignores `split_fraction`
- **File:** `src/hyrax/pytorch_ignite.py`
- Iterates via `iter(provider)` / `next(provider)` and completely bypasses `SubsetRandomSampler`.
- Users defining `split_fraction` on an "infer" group will process **all** data, not the split subset.
- **Fix:** ONNX inference path needs to use the subset-aware dataloader rather than raw iteration.

### 5. Path resolution inconsistency
- **File:** `src/hyrax/config_schemas/data_request.py`, `src/hyrax/data_sets/data_provider.py`
- The Pydantic `resolve_data_location` validator resolves relative paths to absolute.
- `validate_split_fractions_within_location` groups configs by `data_location`.
- If some configs bypass Pydantic validation (e.g., constructed programmatically), resolved vs. unresolved paths won't group together.
- **Fix:** Normalize paths before grouping, or ensure all construction paths go through validation.

---

## 🟡 Medium — Logic Gaps

### 6. Iterable-dataset path hard-codes `split == "train"`
- **File:** `src/hyrax/data_sets/dist_data_loader.py`
- The iterable dataset branch checks `if split == "train"`.
- With the move to arbitrary group names, a custom group like `"fit"` would silently skip the intended shuffle behavior and default to `shuffle=False`.
- **Fix:** Default to `shuffle=True` for any non-infer/non-test group, or make shuffle configurable.

### 7. Legacy path only creates loaders for groups present in `data_request`
- **File:** `src/hyrax/data_sets/data_provider.py`
- In legacy mode (path 3 of `_setup_data`), if a user only defines `"train"` with `split_percentages`, the `"validate"` loader is never created because it's not in `data_request`.
- This is a regression from the old behavior.
- **Fix:** When using percentage-based splitting in legacy mode, auto-create the expected split groups.

### 8. Index allocation rounding loses indices
- **File:** `src/hyrax/data_sets/data_provider.py`
- `_get_split_indices` computes `count = int(total * fraction)` independently per split.
- For `total=10` with fractions `0.33/0.33/0.34`, you get `3+3+3=9` — one index lost.
- The clamping prevents overrun but doesn't redistribute remainders.
- **Fix:** Assign leftover indices to the last split, or distribute round-robin.

### 9. `__getattr__` on `DataRequestDefinition` silently returns wrong thing for `"validate"`
- **File:** `src/hyrax/config_schemas/data_request.py`
- Pydantic's `RootModel` inherits a `validate` classmethod.
- `__getattr__` only fires when normal attribute lookup fails, so `definition.validate` returns Pydantic's method, not the `"validate"` dataset group.
- The docstring warns about this, but existing code using attribute access will silently break.
- **Fix:** Override `__getattr__` to check `root` first, or always use bracket access and lint for attribute access.

### 10. `split_fraction=0.0` produces an empty split
- **File:** `src/hyrax/config_schemas/data_request.py`
- The field allows `ge=0.0`, which creates a `SubsetRandomSampler` with zero indices.
- This will crash downstream when a model tries to get a batch.
- **Fix:** Change constraint to `gt=0.0` to reject zero fractions at validation time.

---

## 🟢 Low — Minor / Testing Gaps

### 11. Global RNG mutation
- **File:** `src/hyrax/data_sets/data_provider.py`
- `_get_split_indices` uses `np.random.seed()` / `np.random.permutation()` which mutates global state.
- Non-reproducible in multi-threaded scenarios.
- **Fix:** Use `np.random.default_rng(seed)` instead.

### 12. Dict iteration order dependency
- **File:** `src/hyrax/data_sets/data_provider.py`
- Split index assignment in `_get_split_indices` depends on iteration order of the fractions dict.
- If the same config is loaded with groups in different order (possible across TOML parsers), different indices get assigned to each split.
- **Fix:** Sort fraction keys before iterating, or document the order dependency.

### 13. No TOML round-trip test
- Tests construct `DataRequestDefinition` directly from dicts.
- There's no test verifying that `split_fraction` written in a `.toml` file survives through parsing → validation → `as_dict()` → `DataProvider`.
- **Fix:** Add an integration test with a real `.toml` file.

### 14. No test for multi-config (dict-of-configs) groups with `split_fraction`
- All tests use single-config groups.
- A group like `{"data_a": config1, "data_b": config2}` where fractions apply differently isn't tested.
- The cross-group `validate_split_fractions_within_location` validator would apply to it.
- **Fix:** Add test cases for multi-config groups.

### 15. `ge=0.0` should probably be `gt=0.0`
- **File:** `src/hyrax/config_schemas/data_request.py`
- A `split_fraction` of exactly `0.0` is almost certainly a user error.
- Should be rejected at validation time rather than creating an empty sampler.
- **Fix:** Change field constraint from `ge=0.0` to `gt=0.0`.

---

## Further Considerations

- **Floating-point tolerance:** Should the Pydantic validator use `math.isclose(total, 1.0)` or simply check `> 1.0 + epsilon`? The latter is simpler and still catches overallocation.
- **Remainder redistribution:** Should `_get_split_indices` assign leftover indices (from rounding) to the last split, or distribute them round-robin?
- **`split_file` support:** The schema defines `split_file` as a concept but it doesn't appear to be wired through `DataProvider` or `dist_data_loader` yet. Is that intentional for a follow-up PR?
