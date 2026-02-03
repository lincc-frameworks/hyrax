# Hyrax — Claude Code Instructions

> **Read `HYRAX_GUIDE.md` first.** It contains the canonical project reference
> (architecture, config system, registries, conventions). This file adds only
> Claude Code-specific guidance.

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
2. Decorate the class with `@hyrax_model`.
3. Implement `__init__`, `forward`, `train_step`, and `prepare_inputs`.
4. Add default config under `[model.YourModelName]` in `hyrax_default_config.toml`.
5. Add tests in `tests/hyrax/`.

### New Dataset
1. Create a file in `src/hyrax/data_sets/`.
2. Subclass `HyraxDataset` (auto-registered via `__init_subclass__`).
3. For image datasets, also inherit `HyraxImageDataset` for transform stacking.
4. Add default config under `[data_set.YourDatasetName]` in `hyrax_default_config.toml`.

### New Verb
1. Create a file in `src/hyrax/verbs/`.
2. Subclass `Verb`, set `cli_name`, decorate with `@hyrax_verb`.
3. Implement `setup_parser`, `run_cli`, and `run`.
4. Verbs are internal only — there is no external verb plugin system.

## Common Pitfalls

- **`key = false` means `None`** — TOML has no null. Hyrax uses `false` as a sentinel
  for "not set." Code must treat `False` as `None` for these keys.
- **`ConfigDict` is Pydantic's** — `from pydantic import ConfigDict`. The runtime
  config is an ordinary mutable `dict`, not a custom wrapper.
- **Verbs are internal only** — external plugins register models and datasets, not verbs.
- **Spelling:** `HyraxCifarDataset` (lowercase 's' in "set"), `HSCDataSet` (capital 'S').
- **Manifest files are a compromise** — they exist out of necessity, not as a design
  goal. Do not treat them as an architectural pattern to extend.
- **Pydantic validation is experimental** — it applies only to the `[data_request]`
  config section. Do not assume Pydantic covers the full config.
