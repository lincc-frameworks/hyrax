# Plan: Move data_request validation to verb classes

## TL;DR
Move cross-group `data_request` validators (split_fraction sum/consistency) from `ConfigManager` to `Verb.__init__`, scoped to only the data groups the verb uses via `REQUIRED_DATA_GROUPS` + `OPTIONAL_DATA_GROUPS`. Keep per-field validation (dataset_class, primary_id_field, etc.) eager at `set_config` time. This fixes #787 where unrelated data groups cause false validation failures.

## Decisions
- **set_config** keeps eager per-field validation (each DataRequestConfig is valid individually) but defers cross-group checks
- **Verb-time validation** raises `RuntimeError` on failure (hard block, not a warning)
- **Verbs without DATA_GROUPS** (umap, search, lookup, etc.) skip data_request validation entirely
- `_validate_data_request` remains in ConfigManager but is modified to only do per-config validation
- `model_inputs` follows the same pattern as `data_request` (it's the deprecated name for the same thing)

---

## Phase 1: Split DataRequestDefinition validation into two layers

**Files:**
- `src/hyrax/config_schemas/data_request.py`

**Steps:**

1. Create a new method on `DataRequestDefinition`: `validate_cross_group(groups: set[str]) -> None`
   - Accepts a set of active group names (the ones the verb will use)
   - Filters `self.root` down to only those groups
   - Runs the split_fraction_sums and split_fraction_consistency logic on the filtered subset
   - Raises `ValueError` on failure (same error messages as current validators)

2. Add a class-level flag or parameter to control whether cross-group validators run during `model_validate()`. The cleanest approach: extract the cross-group validation logic into standalone methods that the `@model_validator` decorators call, AND that `validate_cross_group()` calls on a filtered view. The `@model_validator` versions become no-ops (or are removed), since cross-group validation is now deferred.
   - **Specifically**: Remove `validate_split_fraction_sums` and `validate_split_fraction_consistency` as `@model_validator` methods. Their logic moves into `validate_cross_group()`.
   - Keep `require_at_least_one_dataset` and `validate_primary_id_fields` as `@model_validator` methods — these are per-group checks that should stay eager.

**Relevant functions:**
- `DataRequestDefinition.validate_split_fraction_sums()` — remove as model_validator
- `DataRequestDefinition.validate_split_fraction_consistency()` — remove as model_validator
- `_iter_all_configs()` — reuse in validate_cross_group, but filtered

---

## Phase 2: Add data_request validation to Verb base class

**Files:**
- `src/hyrax/verbs/verb_registry.py`

**Steps:**

3. Add default class attributes to `Verb`:
   ```
   REQUIRED_DATA_GROUPS: tuple[str, ...] = ()
   OPTIONAL_DATA_GROUPS: tuple[str, ...] = ()
   ```

4. Add a `validate_data_request(self)` method to `Verb`:
   - Reads `self.config["data_request"]` (or `self.config["model_inputs"]` for backward compat)
   - If no data_request config exists, return (nothing to validate)
   - If neither REQUIRED_DATA_GROUPS nor OPTIONAL_DATA_GROUPS is defined on the subclass, return (skip validation)
   - Compute `active_groups = set(REQUIRED_DATA_GROUPS + OPTIONAL_DATA_GROUPS) & set(data_request.keys())`
   - Validate that all REQUIRED_DATA_GROUPS are present in data_request keys; raise RuntimeError if missing
   - Build a DataRequestDefinition from the full data_request dict
   - Call `definition.validate_cross_group(active_groups)` — raises ValueError on failure
   - Wrap any ValueError in a RuntimeError with a user-friendly message

5. Call `self.validate_data_request()` at the end of `Verb.__init__()`, after `self.config = config`. This means validation runs automatically when any verb is instantiated (which happens in `Hyrax.__getattr__`).

---

## Phase 3: Remove cross-group validation from ConfigManager

**Files:**
- `src/hyrax/config_utils.py`

**Steps:**

6. Modify `ConfigManager._validate_data_request()`:
   - Still calls `DataRequestDefinition.model_validate(value)` — this handles normalization and per-field validation
   - Cross-group validators no longer fire (removed from model in Phase 1)
   - Returns the validated dict as before

7. No changes needed to `set_config()` or `__init__()` call sites — they still call `_validate_data_request`, which now only does per-field checks. The warning-on-failure behavior stays intact.

---

## Phase 4: Update tests

**Files:**
- `tests/hyrax/test_data_request_config.py`
- `tests/hyrax/test_config_validation_warnings.py`
- `tests/hyrax/test_verb_data_request_validation.py` (new)

**Steps:**

8. Update `tests/hyrax/test_data_request_config.py`:
   - Tests for `validate_split_fraction_sums` and `validate_split_fraction_consistency` that call `DataRequestDefinition.model_validate()` directly should be updated: they should no longer raise during construction. Instead, test the new `validate_cross_group()` method.
   - Tests for per-field validation (`primary_id_field`, `split_fraction` range, `require_primary_id_for_split_fraction`) remain unchanged.
   - Tests for `require_at_least_one_dataset` remain unchanged.

9. Update `tests/hyrax/test_config_validation_warnings.py`:
   - Tests that expect warnings about split_fraction consistency at `set_config` time need updating (if any). Most existing tests fail on `primary_id_field` which is still an eager check, so they should be fine.

10. Create `tests/hyrax/test_verb_data_request_validation.py`:
    - Test that `Verb.validate_data_request()` filters data groups correctly
    - Test the exact scenario from issue #787: train+validate with split_fraction + infer without split_fraction — Train verb should validate only train+validate groups (passes)
    - Test that missing REQUIRED_DATA_GROUPS raises RuntimeError
    - Test that verbs without DATA_GROUPS skip validation
    - Test that invalid cross-group config for active groups raises RuntimeError
    - Can use `Train`, `Infer`, `Test` verbs directly with mock configs

---

## Relevant files

- `src/hyrax/config_schemas/data_request.py` — remove cross-group `@model_validator` methods, add `validate_cross_group(groups)`
- `src/hyrax/verbs/verb_registry.py` — add `REQUIRED/OPTIONAL_DATA_GROUPS` defaults and `validate_data_request()` to `Verb`
- `src/hyrax/config_utils.py` — `_validate_data_request()` unchanged (cross-group validators just won't fire anymore)
- `src/hyrax/verbs/train.py` — no changes needed (already defines DATA_GROUPS)
- `src/hyrax/verbs/infer.py` — no changes needed
- `src/hyrax/verbs/test.py` — no changes needed
- `src/hyrax/verbs/visualize.py` — can remove ad-hoc RuntimeError check in `run()` (lines 122-128) since base class now validates
- `tests/hyrax/test_data_request_config.py` — update cross-group tests
- `tests/hyrax/test_config_validation_warnings.py` — verify existing tests still pass
- `tests/hyrax/test_verb_data_request_validation.py` — new test file for verb-time validation

## Verification

1. `ruff check src/ tests/ && ruff format src/ tests/`
2. `python -m pytest tests/hyrax/test_data_request_config.py tests/hyrax/test_config_validation_warnings.py tests/hyrax/test_verb_data_request_validation.py -v` — targeted test run
3. `python -m pytest -m "not slow"` — full fast suite
4. Manual verification: reproduce the exact scenario from issue #787 in a Python shell:
   - Create data_request with train (split 0.8) + validate (split 0.2) + infer (no split) → set_config should succeed
   - Instantiate Train verb → should pass (only validates train+validate)
   - Instantiate Infer verb → should pass (only validates infer)
5. `pre-commit run --all-files`

## Further Considerations

1. **model_inputs backward compat**: The `validate_data_request` method in Verb should check both `data_request` and `model_inputs` config keys — recommend checking `data_request` first, falling back to `model_inputs`.
2. **Serialization round-trip**: After removing cross-group validators from the Pydantic model, configs that previously failed at set_config time will now succeed and be normalized. This changes observable behavior (the warning disappears for the #787 scenario) — this is the desired fix, but worth noting.
