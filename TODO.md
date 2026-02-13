# Edge Cases for Split-Fraction Branch (`issue/663/define-split-fraction-in-data-request`)

PR #690: Updated `data_request` to support splits

---

## 🔴 High — Likely Bugs or Silent Data Corruption

### 1. ✅ FIXED - Floating-point sum in `validate_split_fractions_within_location`
- **File:** `src/hyrax/config_schemas/data_request.py`
- ~~Uses exact `> 1.0` comparison with no epsilon.~~
- ~~Classic FP issue: `0.1 + 0.2 + 0.7 = 1.0000000000000002` → rejects a valid config.~~
- ~~The runtime counterpart in `_get_split_indices` uses `min(end, total)` clamping, but the Pydantic validator will reject the config before it ever gets there.~~
- **Fix Applied:** Changed to `np.round(total, decimals=5) > 1.0` to match pytorch_ignite.py approach.
- **Test Added:** `test_split_fraction_sum_fp_rounding_at_one_passes`

### 2. ✅ FIXED - Provider length validation in `create_splits_from_fractions`
- **File:** `src/hyrax/pytorch_ignite.py`
- ~~If providers for the same `data_location` have different lengths (e.g., different filter criteria or caching issues), split indices will silently point out-of-range or at wrong items in shorter providers.~~
- ~~No assertion verifies uniform `len()` across providers.~~
- **Fix Applied:** Added explicit length validation in `create_splits_from_fractions` with clear comment explaining the check.
- **Test Added:** `test_error_when_provider_lengths_mismatch`

### 3. ✅ FIXED - `shuffle=True` + `SubsetRandomSampler` conflict
- **File:** `src/hyrax/pytorch_ignite.py` (dist_data_loader function, lines 268-286)
- ~~When `split_fraction` triggers a `SubsetRandomSampler`, PyTorch's `DataLoader` raises if both `shuffle=True` and a sampler are provided.~~
- ~~The `train` and `infer` verbs guard against this, but `dist_data_loader` itself does **not** guard on the code paths where it passes `shuffle` through.~~
- **Fix Applied:** In `dist_data_loader`, force `shuffle=False` in a copy of kwargs when a sampler is present (lines 278-286).
- **Test Added:** `test_shuffle_true_with_split_indices_does_not_error`

### 4. ✅ FIXED - ONNX inference ignores `split_fraction`
- **File:** `src/hyrax/verbs/engine.py`
- ~~Iterates via `iter(provider)` / `next(provider)` and completely bypasses `SubsetRandomSampler`.~~
- ~~Users defining `split_fraction` on an "infer" group will process **all** data, not the split subset.~~
- **Fix Applied:** Engine verb now checks if the dataset has `split_indices` set and uses those for iteration instead of `range(0, len(infer_dataset))` (lines 120-130).
- **Test Added:** `TestEngineSplitIndices` with `test_engine_respects_split_indices` and `test_engine_processes_all_indices_when_no_split_indices`

### 5. ❌ WON'T DO - Path resolution inconsistency
- **File:** `src/hyrax/config_schemas/data_request.py`, `src/hyrax/data_sets/data_provider.py`
- ~~The Pydantic `resolve_data_location` validator resolves relative paths to absolute.~~
- ~~`validate_split_fractions_within_location` groups configs by `data_location`.~~
- ~~If some configs bypass Pydantic validation (e.g., constructed programmatically), resolved vs. unresolved paths won't group together.~~
- **Analysis:** Not a real issue. The `@field_validator("data_location")` runs before `@model_validator(mode="after")`, so all paths are already normalized by `resolve_data_location` before `validate_split_fraction_sums` groups them. The only way to bypass this is direct programmatic construction that also bypasses validation entirely.

---

## 🟡 Medium — Logic Gaps

### 6. ❌ WON'T DO - Iterable-dataset path hard-codes `split == "train"`
- **File:** `src/hyrax/data_sets/dist_data_loader.py`
- ~~The iterable dataset branch checks `if split == "train"`.~~
- ~~With the move to arbitrary group names, a custom group like `"fit"` would silently skip the intended shuffle behavior and default to `shuffle=False`.~~
- **Reason:** All iterable-dataset code will be removed in a different PR that will be merged shortly.

### 7. ❌ WON'T DO - Legacy path only creates loaders for groups present in `data_request`
- **File:** `src/hyrax/data_sets/data_provider.py`
- ~~In legacy mode (path 3 of `_setup_data`), if a user only defines `"train"` with `split_percentages`, the `"validate"` loader is never created because it's not in `data_request`.~~
- ~~This is a regression from the old behavior.~~
- **Reason:** Avoiding changes to legacy code path to prevent regressions affecting backward compatibility or existing legacy users.

### 8. ✅ FIXED - Index allocation rounding loses indices
- **File:** `src/hyrax/pytorch_ignite.py` (create_splits_from_fractions function)
- ~~`_get_split_indices` computes `count = int(total * fraction)` independently per split.~~
- ~~For `total=10` with fractions `0.33/0.33/0.34`, you get `3+3+3=9` — one index lost.~~
- ~~The clamping prevents overrun but doesn't redistribute remainders.~~
- **Fix Applied:** When fractions sum to ~1.0, leftover indices from rounding are now assigned to the last split. When fractions sum to <1.0, indices remain unassigned as intended (lines 510-527).
- **Test Added:** `test_rounding_leftover_indices_assigned_to_last_split`

### 9. ✅ FIXED - `__getattr__` on `DataRequestDefinition` collision with `validate`
- **File:** `src/hyrax/config_schemas/data_request.py`
- ~~Pydantic's `RootModel` inherits a `validate` classmethod.~~
- ~~`__getattr__` only fires when normal attribute lookup fails, so `definition.validate` returns Pydantic's method, not the `"validate"` dataset group.~~
- ~~The docstring warns about this, but existing code using attribute access will silently break.~~
- **Fix Applied:** Removed `__getattr__` entirely and standardized on bracket-style access (`__getitem__`) throughout the codebase. Updated all tests to use `definition["train"]` instead of `definition.train`.
- **Reason:** Eliminates the validate collision issue, provides a consistent access pattern, and makes the dictionary-like interface explicit.

### 10. ✅ FIXED - `split_fraction=0.0` produces an empty split
- **File:** `src/hyrax/config_schemas/data_request.py`
- ~~The field allows `ge=0.0`, which creates a `SubsetRandomSampler` with zero indices.~~
- ~~This will crash downstream when a model tries to get a batch.~~
- **Fix Applied:** Changed constraint from `ge=0.0` to `gt=0.0` to reject zero fractions at validation time.
- **Test Updated:** `test_split_fraction_invalid_values` now includes 0.0 as an invalid value; `test_split_fraction_valid_values` no longer tests 0.0.

---

## 🟢 Low — Minor / Testing Gaps

### 11. ❌ WON'T DO - Global RNG mutation
- **File:** `src/hyrax/data_sets/data_provider.py`
- ~~`_get_split_indices` uses `np.random.seed()` / `np.random.permutation()` which mutates global state.~~
- ~~Non-reproducible in multi-threaded scenarios.~~
- **Reason:** The global RNG mutation is isolated to the split creation phase, which happens once during dataset setup. Multi-threaded training/inference uses DataLoader workers which have separate RNG states. Not worth the refactoring risk for minimal benefit.

### 12. ✅ FIXED - Dict iteration order dependency
- **File:** `src/hyrax/pytorch_ignite.py` (create_splits_from_fractions function)
- ~~Split index assignment in `_get_split_indices` depends on iteration order of the fractions dict.~~
- ~~If the same config is loaded with groups in different order (possible across TOML parsers), different indices get assigned to each split.~~
- **Fix Applied:** Added documentation clarifying that iteration order preserves the dict insertion order, which comes from the `splits` parameter in `setup_dataset()`. Since Python 3.7+ guarantees dict insertion order, and TOML 1.0 maintains table order, the split assignment is deterministic and matches the order specified in the splits parameter.

### 13. ❌ WON'T DO - No TOML round-trip test
- ~~Tests construct `DataRequestDefinition` directly from dicts.~~
- ~~There's no test verifying that `split_fraction` written in a `.toml` file survives through parsing → validation → `as_dict()` → `DataProvider`.~~
- **Reason:** Existing integration tests exercise the full config loading pipeline from TOML through to DataProvider. Adding a specific round-trip test provides minimal additional coverage and would duplicate existing test patterns.

### 14. ❌ WON'T DO - No test for multi-config (dict-of-configs) groups with `split_fraction`
- ~~All tests use single-config groups.~~
- ~~A group like `{"data_a": config1, "data_b": config2}` where fractions apply differently isn't tested.~~
- ~~The cross-group `validate_split_fractions_within_location` validator would apply to it.~~
- **Reason:** Multi-config groups (dict-of-configs within a single group) is an advanced/rare use case. The validator logic is straightforward and tested via single-config groups. Adding multi-config tests would increase maintenance burden without proportional benefit.

### 15. ✅ FIXED - `ge=0.0` should probably be `gt=0.0` (Duplicate of Item 10)
- **File:** `src/hyrax/config_schemas/data_request.py`
- ~~A `split_fraction` of exactly `0.0` is almost certainly a user error.~~
- ~~Should be rejected at validation time rather than creating an empty sampler.~~
- **Fix Applied:** Resolved in item 10 - changed field constraint from `ge=0.0` to `gt=0.0`.

---

## Further Considerations

- ✅ **DONE - Floating-point tolerance:** Resolved in item 1. The Pydantic validator uses `np.round(total, decimals=5) > 1.0` which provides 5 decimal places of tolerance - simpler than `math.isclose()` and still catches overallocation.
- ✅ **DONE - Remainder redistribution:** Resolved in item 8. `_get_split_indices` assigns leftover indices (from rounding) to the last split when fractions sum to ~1.0, using tolerance check `total >= 1.0 - 1e-5`.
- 🔮 **FUTURE WORK - `split_file` support:** The schema defines `split_file` as a concept but it doesn't appear to be wired through `DataProvider` or `dist_data_loader` yet. This will be implemented in a future PR.

---

## Additional Enhancements

### ✅ Enhanced Deprecation Warning for Legacy Split Configuration
- **File:** `src/hyrax/pytorch_ignite.py` (create_splits function)
- **Enhancement:** Added a prominent, multi-line `FutureWarning` at the start of the `create_splits` function to alert users that the legacy configuration style using `config["data_set"]["train_size"]`, `config["data_set"]["validate_size"]`, and `config["data_set"]["test_size"]` is deprecated.
- **Details:** The warning includes:
  - Clear deprecation notice with visual separators (80-character border)
  - Migration instructions with step-by-step guidance
  - Side-by-side OLD vs NEW configuration examples
  - Link to documentation for further assistance
- **Rationale:** Makes the deprecation highly visible and provides actionable guidance for users to migrate to the new `split_fraction` approach in `data_request` configuration.
