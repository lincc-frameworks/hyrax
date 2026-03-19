# Plan: Checkpoint on validation loss when validator exists

## TL;DR
Issue #762: `best_checkpoint` currently scores on **training batch loss** (`trainer.state.output["loss"]`). When a validator is present, it should score on **validation loss** instead. When no validator exists, behavior is unchanged.

**Approach:** Extract best-checkpoint logic into a new standalone function `attach_best_checkpoint()` in `pytorch_ignite.py`. Call it from `train.py`, passing either the validator or trainer as the scoring engine. No changes to `create_trainer` or `create_validator` return types or signatures.

## Critical Discovery

**Event ordering context:** In `create_trainer` (called first), both checkpoint handlers are registered on `HYRAX_EPOCH_COMPLETED`. In `create_validator` (called second), `run_validation` is *also* registered on the **trainer's** `HYRAX_EPOCH_COMPLETED`. Since Ignite fires handlers in registration order, checkpoints currently fire **before** validation runs. By moving best-checkpoint registration to happen *after* `create_validator` sets up its `run_validation` handler, and attaching it to the **validator's** event, both ordering and scoring source are correct.

The existing code already has a comment anticipating this refactor:
```
#! We may want to move the checkpointing logic over to the `validator`.
```

## Decisions
- **No return type changes:** `create_trainer` returns `Engine`, `create_validator` returns `Engine`.
- **No new parameters** on `create_trainer` or `create_validator`.
- **Filename pattern:** Same naming regardless of what drives the checkpoint.
- **`latest_checkpoint`:** Stays on the trainer always. Only `best_checkpoint` moves.
- **Checkpoint file format:** Unchanged `to_save` dict — resume is fully backward-compatible.
- **`auto_model` double-call:** The new function calls `idist.auto_model(model)` to build `to_save`. This is safe: in non-distributed mode it returns the same reference; in distributed mode, multiple DDP wrappers share the underlying module so state_dict serialization is equivalent. (The codebase already calls `auto_model` separately in both `create_trainer` and `create_validator`.)

## Steps

### Phase 1: New function in `pytorch_ignite.py`

1. **Create `attach_best_checkpoint(engine, model, trainer, results_directory)`** — a new public function that:
   - Builds `to_save` dict: `{"model": idist.auto_model(model), "optimizer": model.optimizer, "trainer": trainer}` + scheduler if present.
   - Creates `neg_loss_score` (reads `engine.state.output["loss"]`).
   - Creates `Checkpoint` with `DiskSaver(results_directory)`, `n_saved=1`, `global_step_transform=global_step_from_engine(trainer)`, `score_name="loss"`, `score_function=neg_loss_score`, `greater_or_equal=True`.
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

### Phase 3: Update `Train.run()` in `verbs/train.py`

4. **Add import** of `attach_best_checkpoint` from `hyrax.pytorch_ignite`.
5. **After** the existing `create_validator` call block, add checkpointing orchestration:
   - If `validation_data_loader is not None` (validator was created): call `attach_best_checkpoint(validator, model, trainer, results_dir)` where `validator` is the return value of `create_validator`.
   - Else: call `attach_best_checkpoint(trainer, model, trainer, results_dir)`.
6. This requires capturing the `validator` return value. Currently `create_validator(...)` is called but its return is discarded. Change to: `validator = create_validator(...)`.

### Phase 4: Tests in `tests/hyrax/test_train.py`

7. **New test: `test_best_checkpoint_uses_validation_loss`** — Train with a validation split for 2+ epochs. Inspect results directory for `best_checkpoint` file. Verify it exists and the training completed successfully. (The loopback model's `validate_batch` returns `{"loss": 0.0}`, which is a valid score for checkpoint selection.)
8. **New test: `test_best_checkpoint_without_validation`** — Train without validation split. Verify `best_checkpoint` file still appears in results directory (existing behavior preserved).
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
