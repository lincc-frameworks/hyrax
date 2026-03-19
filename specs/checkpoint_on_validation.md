# Plan: Checkpoint on validation loss when validator exists

## TL;DR
Issue #762: `best_checkpoint` currently scores on **training batch loss** (`trainer.state.output["loss"]`). When a validator is present, it should score on **validation loss** instead. When no validator exists, behavior is unchanged.

**Approach:** Extract best-checkpoint logic into a new standalone function `attach_best_checkpoint()` in `pytorch_ignite.py`. Call it from `train.py`, passing either the validator or trainer as the scoring engine. No changes to `create_trainer` or `create_validator` return types or signatures. Each engine factory sets a `hyrax_label` attribute on the returned `Engine` so callers do not need to pass a separate type string.

## Critical Discovery

**Event ordering context:** In `create_trainer` (called first), both checkpoint handlers are registered on `HYRAX_EPOCH_COMPLETED`. In `create_validator` (called second), `run_validation` is *also* registered on the **trainer's** `HYRAX_EPOCH_COMPLETED`. Since Ignite fires handlers in registration order, checkpoints currently fire **before** validation runs. By moving best-checkpoint registration to happen *after* `create_validator` sets up its `run_validation` handler, and attaching it to the **validator's** event, both ordering and scoring source are correct.

The existing code already has a comment anticipating this refactor:
```
#! We may want to move the checkpointing logic over to the `validator`.
```

## Decisions
- **No return type changes:** `create_trainer` returns `Engine`, `create_validator` returns `Engine`.
- **No new parameters** on `create_trainer` or `create_validator`.
- **`hyrax_label` attribute:** Each engine factory (`create_trainer`, `create_validator`, `create_evaluator`, `create_tester`) sets `engine.hyrax_label` on the returned `Engine` (e.g. `"trainer"`, `"validator"`, `"evaluator"`, `"tester"`). This is a Hyrax-specific attribute; `ignite.Engine` does not define a `label` attribute of its own.
- **Filename pattern:** The best-checkpoint filename embeds `engine.hyrax_label` as the score name prefix (e.g. `validator_loss=…` or `trainer_loss=…`). `attach_best_checkpoint` reads `engine.hyrax_label` via `hasattr` and falls back to a plain `"loss"` score name if the attribute is absent.
- **`latest_checkpoint`:** Stays on the trainer always. Only `best_checkpoint` moves.
- **Checkpoint file format:** Unchanged `to_save` dict — resume is fully backward-compatible.
- **`auto_model` double-call:** The new function calls `idist.auto_model(model)` to build `to_save`. This is safe: in non-distributed mode it returns the same reference; in distributed mode, multiple DDP wrappers share the underlying module so state_dict serialization is equivalent. (The codebase already calls `auto_model` separately in both `create_trainer` and `create_validator`.)

## Steps

### Phase 1: New function in `pytorch_ignite.py`

1. **Create `attach_best_checkpoint(engine, model, trainer, results_directory)`** — a new public function that:
   - Builds `to_save` dict: `{"model": idist.auto_model(model), "optimizer": model.optimizer, "trainer": trainer}` + scheduler if present.
   - Creates `neg_loss_score` (reads `engine.state.output["loss"]`).
   - Reads `engine.hyrax_label` (via `hasattr`) to build the `score_name` (e.g. `"validator_loss"` or `"trainer_loss"`); falls back to `"loss"` if absent.
   - Creates `Checkpoint` with `DiskSaver(results_directory)`, `n_saved=1`, `global_step_transform=global_step_from_engine(trainer)`, `score_name=<label>_loss`, `score_function=neg_loss_score`, `greater_or_equal=True`.
   - Registers checkpoint on `engine`'s `HyraxEvents.HYRAX_EPOCH_COMPLETED`.
   - Registers `log_best_checkpoint_location` on `trainer`'s `Events.COMPLETED`.
   - `engine` is the engine whose output loss drives the score — the validator when it exists, the trainer otherwise.

### Phase 2: Remove best-checkpoint from `create_trainer`

2. **Remove** from `create_trainer`:
   - The `best_checkpoint` `Checkpoint` creation block (the `neg_loss_score` closure, the `Checkpoint(...)` call with `score_function`).
   - `trainer.add_event_handler(HyraxEvents.HYRAX_EPOCH_COMPLETED, best_checkpoint)`.
   - `log_best_checkpoint_location` closure and its `trainer.add_event_handler(Events.COMPLETED, ...)` registration.
   - The anticipatory comment (`#! We may want to move...`).
3. **Keep** in `create_trainer` (unchanged):
   - `to_save` dict construction (needed for `latest_checkpoint` and resume).
   - `latest_checkpoint` creation and registration on `HYRAX_EPOCH_COMPLETED`.
   - `log_last_checkpoint_location` and its registration on `Events.COMPLETED`.
   - All other handlers, resume logic, progress bar.
   - Return type remains `Engine`.
   - **Set `trainer.hyrax_label = "trainer"`** before returning.

### Phase 3: Update `Train.run()` in `verbs/train.py`

4. **Add import** of `attach_best_checkpoint` from `hyrax.pytorch_ignite`.
5. **After** the existing `create_validator` call block, add checkpointing orchestration:
   - If `validation_data_loader is not None` (validator was created): call `attach_best_checkpoint(validator, model, trainer, results_dir)` where `validator` is the return value of `create_validator`.
   - Else: call `attach_best_checkpoint(trainer, model, trainer, results_dir)`.
6. This requires capturing the `validator` return value. Currently `create_validator(...)` is called but its return is discarded. Change to: `validator = create_validator(...)`.

### Phase 4: Tests in `tests/hyrax/test_train.py`

7. **New test: `test_best_checkpoint_uses_validation_loss`** — Train with a validation split for 2+ epochs. Inspect results directory for a `*val_loss=*.pt` best-checkpoint file. Assert exactly one such file exists, confirming the validator engine drove checkpoint scoring.
8. **New test: `test_best_checkpoint_without_validation`** — Train without validation split. Inspect results directory for a `*trn_loss=*.pt` best-checkpoint file. Assert exactly one such file exists, confirming the trainer engine drove checkpoint scoring.
9. **Existing tests** should pass unchanged — checkpoint file format and resume mechanism are identical.

## Relevant Files
- `src/hyrax/pytorch_ignite.py` — `create_trainer` (L747–882): remove best_checkpoint; new `attach_best_checkpoint` function
- `src/hyrax/verbs/train.py` — `Train.run()` (L165–175): capture validator, call `attach_best_checkpoint`
- `tests/hyrax/test_train.py` — new checkpoint tests
- `tests/hyrax/conftest.py` — `loopback_hyrax` fixture (already configures validation split)

## Verification

1. `ruff check src/ tests/ && ruff format src/ tests/`
2. `python -m pytest tests/hyrax/test_train.py -v` — all existing + new train tests pass
3. `python -m pytest -m "not slow"` — full fast suite passes
4. Manual: train with validation → inspect results dir → `best_checkpoint` scored on validation loss
5. Manual: train without validation → `best_checkpoint` behaves as before
6. `pre-commit run --all-files`
