# Hyrax — Claude Code Instructions

> **`HYRAX_GUIDE.md` is the canonical reference** for architecture, config system,
> registries, and conventions. This file contains only Claude Code-specific overrides.
> When in doubt, follow `HYRAX_GUIDE.md`.

## Project Overview

Hyrax is a low-code ML platform for astronomy built on PyTorch. Key workflows:
download survey cutouts → train models → infer latent representations → visualize.
Primary interface is Jupyter notebooks; the `hyrax` CLI is secondary (HPC/Slurm).

## Development Workflow

- **Style:** Run `ruff check src/ tests/ && ruff format src/ tests/` and let the linter
  fix issues. Do not hand-tune formatting.
- **Tests:** `python -m pytest -m "not slow"` for the fast suite.
- **Pre-commit:** `pre-commit run --all-files` before finishing work.
- **Docs:** `sphinx-build -M html ./docs ./_readthedocs`

## Code Style

- Python ≥ 3.11 — use modern syntax (`match`, `X | Y` unions, etc.).
- Line length: 110 (enforced by ruff).
- Rely on the linter — do not manually reformat surrounding code.
- Add docstrings to new public functions; follow existing NumPy-style conventions.
- Never commit code containing the note-to-self marker (`xc` repeated twice).
  A pre-commit hook will block the commit.

## Adding Components

### New Model
1. Create a file in `src/hyrax/models/`.
2. Inherit from `torch.nn.Module`, decorate the class with `@hyrax_model`.
3. Implement `__init__`, `forward`, `train_step`, and `prepare_inputs`.
4. Add default config under `[model.YourModelName]` in `hyrax_default_config.toml`.
5. Add tests in `tests/hyrax/test_<name>.py`.

### New Dataset
1. Create a file in `src/hyrax/data_sets/`.
2. Subclass `HyraxDataset` (auto-registered via `__init_subclass__`). Use `Dataset` spelling.
3. For image datasets, also inherit `HyraxImageDataset` for transform stacking.
4. Add default config under `[data_set.YourDatasetName]` in `hyrax_default_config.toml`.

### New Verb
1. Create a file in `src/hyrax/verbs/`.
2. Subclass `Verb`, set `cli_name`, decorate with `@hyrax_verb`.
3. Implement `setup_parser`, `run_cli`, and `run`.
4. Verbs are internal only. Always use class-based verbs (function-based verbs are legacy).

## Common Pitfalls

See `HYRAX_GUIDE.md` for the full list. Key points:

- **`key = false` means `None`** — TOML has no null. Hyrax uses `false` as a sentinel
  for "not set." Code must treat `False` as `None` for these keys.
- **`ConfigDict` is Pydantic's** — `from pydantic import ConfigDict`. The runtime
  config is an ordinary mutable `dict`, not a custom wrapper.
- **Verbs are internal only** — external plugins register models and datasets, not verbs.
- **Manifest files** — ask the user before extending this pattern.
- **Pydantic validation** — do not add to new config sections.
