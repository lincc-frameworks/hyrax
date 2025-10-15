---
description: "Task list for flexible dataset splitting feature"
---

# Tasks: Flexible Dataset Splitting for Model Training

**Input**: Design documents from `/specs/001-build-a-flexible/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 Create split mode abstraction (registry/factory pattern) in codebase
- [ ] T002 Update TOML config parsing to support nested split definitions under `[model_inputs.<friendly_name>.<split>]` (e.g., `[model_inputs.data.train]`, `[model_inputs.data.validate]`)
- [ ] T003 Ensure backward compatibility with existing config and workflows

---

## Phase 2: Core Implementation

- [ ] T101 Refactor dataset and dataloader setup in `pytorch_ignite.py` to use split mode abstraction
- [ ] T102 Implement logic for "all data in one location" split mode (reuse existing)
- [ ] T103 Implement logic for "data in separate directories" split mode
- [ ] T104 Validate configuration and provide clear error messages for invalid setups
- [ ] T105 Support multi-dataset input via `model_inputs` table

---

## Phase 3: Testing & Validation

- [ ] T201 Add unit tests for both split modes
- [ ] T202 Add integration tests for multi-dataset input and edge cases
- [ ] T203 Test performance and usability for new split modes

---

## Phase 4: Documentation & Migration

- [ ] T301 Update documentation and example config files for new split mode options
- [ ] T302 Write migration guide for users

---

## Phase 5: Review & Finalization

- [ ] T401 Code review and refactor for extensibility
- [ ] T402 Final validation against specification and checklist
- [ ] T403 Announce feature and collect user feedback
