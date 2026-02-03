# Hyrax Guide

Canonical reference for AI coding assistants working on Hyrax. Both `CLAUDE.md` and
`.github/copilot-instructions.md` point here—keep this file as the single source of truth.

## What Is Hyrax

Hyrax is a low-code, model-agnostic platform for machine learning on large astronomical
imaging surveys. It is built on PyTorch and PyTorch Ignite and handles the boilerplate
around downloading cutouts, building latent representations, interactive visualization,
and anomaly detection so astronomers can focus on science.

## Design Philosophy

**"Configuration OR Code"** — Hyrax uses a deliberate three-tier system:

1. **Invisible** — sensible defaults handle it; the user never thinks about it.
2. **Config value** — the user sets a TOML key and Hyrax does the rest.
3. **Write code** — when the config system is not enough, the user writes a class or
   function and points the config at it via an import path.

**Jupyter notebooks are the primary interface.** The CLI (`hyrax` command) is the
secondary interface, intended for HPC / Slurm batch jobs. The CLI should be able to do
everything notebooks can.

**Results directories are the backbone of reproducibility.** Each run creates a
timestamped directory (`YYYYMMDD-HHMMSS-<verb>-<uid>`) under `results/` containing
model weights, config snapshots, and MLflow tracking data. MLflow is one piece of this
system, not the whole story.

**Target scale:** 10M–100M objects on a unix filesystem.

**Document current behavior.** When migrating away from old patterns, use clear error
messages to guide users rather than silently supporting legacy behavior.

**No changelogs** — use Git history.

## Development Setup

- **Python ≥ 3.11** (see `pyproject.toml` `requires-python`)
- Create a conda env: `conda create -n hyrax python=3.11 && conda activate hyrax`
- Clone and install: `git clone https://github.com/lincc-frameworks/hyrax.git && cd hyrax`
- Run the setup script: `echo 'y' | bash .setup_dev.sh`
  - Installs the package in editable mode with dev extras
  - Installs pre-commit hooks
- Alternative manual install:
  ```
  pip install -e .'[dev]'
  pre-commit install
  ```

## Common Commands

```bash
# Fast tests (default suite)
python -m pytest -m "not slow"

# Slow / integration tests
python -m pytest -m "slow"

# All tests
python -m pytest

# Parallel tests
python -m pytest -n auto

# Lint and format (let the linter fix style — do not hand-tune)
ruff check src/ tests/
ruff format src/ tests/

# Pre-commit (runs ruff, mypy stubs, trailing whitespace, etc.)
pre-commit run --all-files

# Build docs
sphinx-build -M html ./docs ./_readthedocs
```

## Repository Structure

```
src/hyrax/              Main package
src/hyrax/models/       Model definitions and MODEL_REGISTRY
src/hyrax/data_sets/    Dataset implementations and DATASET_REGISTRY
src/hyrax/verbs/        CLI verb implementations and VERB_REGISTRY
src/hyrax/config_schemas/ Pydantic schemas (experimental, data_request only)
src/hyrax/vector_dbs/   ChromaDB / Qdrant integrations
src/hyrax/downloadCutout/ Cutout downloading utilities
src/hyrax_cli/          CLI entry point (main.py)
tests/hyrax/            Test suite
docs/                   Sphinx documentation sources
example_notebooks/      Jupyter notebook examples
benchmarks/             ASV performance benchmarks
```

Key files:

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies, ruff/pytest config |
| `src/hyrax/hyrax_default_config.toml` | Default configuration template |
| `src/hyrax/hyrax.py` | Main `Hyrax` class — config management, verb dispatch |
| `src/hyrax/config_utils.py` | `ConfigManager`, config merging, results directory creation |
| `src/hyrax/plugin_utils.py` | Dynamic class loading for external plugins |
| `.setup_dev.sh` | Development environment bootstrap |

## Architecture: Plugin Registries

Hyrax discovers components through three registries:

### MODEL_REGISTRY (`src/hyrax/models/model_registry.py`)
- Decorator: `@hyrax_model`
- Models must implement `__init__`, `forward`, `train_step`, and `prepare_inputs`.
- The decorator wires up save/load, optimizer, and criterion handling.
- Built-in: `HyraxAutoencoder`, `HyraxAutoencoderV2`, `HyraxCNN`, `SimCLR`, `ImageDCAE`
- **External plugins supported** — use a fully qualified import path in the config
  (e.g. `model.name = "my_pkg.my_module.MyModel"`).

### DATASET_REGISTRY (`src/hyrax/data_sets/data_set_registry.py`)
- Registration: automatic via `HyraxDataset.__init_subclass__`
- Built-in: `HyraxCifarDataset`, `HSCDataSet`, `LSSTDataset`, `FitsImageDataSet`
- **External plugins supported** — same import-path mechanism as models.

### VERB_REGISTRY (`src/hyrax/verbs/verb_registry.py`)
- Decorator: `@hyrax_verb`; base class: `Verb`
- Class-based verbs implement `setup_parser`, `run_cli`, and `run`.
- Some verbs (download, prepare, rebuild_manifest) are function-based in `hyrax.py`.
- **Verbs are internal only** — there is no public plugin system for external verb
  registration. External extensions register through models and datasets only.

## Configuration System

Configuration is TOML-based. Resolution order:

1. Explicit file via `--runtime-config` / `-c` (CLI) or `config_file` parameter (API)
2. `hyrax_config.toml` in the current working directory
3. Packaged `hyrax_default_config.toml`

`ConfigManager` deep-merges user config over defaults (including external library
defaults discovered automatically). The **runtime config is a plain mutable dict** —
code reads and writes it freely at runtime via `ConfigManager.set_config()`.

### `key = false` convention

TOML has no `None`. Hyrax uses `false` as a sentinel meaning "not set / use default
behavior." Code that reads these keys must treat the boolean `False` as `None`.

### Pydantic validation

Pydantic validation is **experimental and limited to the `[data_request]` section only**
(`config_schemas/data_request.py`). The rest of the config is validated by checking keys
against defaults, not by Pydantic schemas. Do not assume Pydantic covers the whole
config.

Note: `ConfigDict` appearing in `config_schemas/` is **Pydantic's `ConfigDict`**, not a
custom Hyrax wrapper. The runtime config itself is an ordinary `dict`.

## Data Flow

High-level pipeline:

1. **Download** — fetch cutouts from survey services; track progress in a manifest file.
2. **Prepare** — apply transforms, build dataset splits (train / validate / test).
3. **Train** — fit a model; results written to a timestamped results directory.
4. **Infer** — run a trained model over a dataset; save latent representations.
5. **UMAP** — reduce dimensionality of latent vectors for visualization.
6. **Visualize** — interactive exploration in Jupyter (holoviews / bokeh).
7. **Vector DB** — store and query latent vectors (ChromaDB or Qdrant).

Each verb that produces output creates its own timestamped results directory.

## Testing Conventions

- **Markers:** `slow` for integration / E2E tests; unmarked tests are fast.
- Default test run: `python -m pytest -m "not slow"`
- Test data: `HyraxCifarDataset` (CIFAR-10 via torchvision), `HyraxRandomDataset`,
  and Pooch/Zenodo-hosted files for slow tests.
- Parallel execution: `pytest -n auto` (pytest-xdist).
- E2E tests exercise full pipelines (train → infer → umap → visualize).

## Key Conventions

- **Timestamped results dirs** — `YYYYMMDD-HHMMSS-<verb>-<uid>` under `results/`.
  Each run snapshots its config as `runtime_config.toml` inside the directory.
- **Batch indexing** — data loaders use PyTorch's standard batch dimension (dim 0).
- **Transform stacking** — `HyraxImageDataset` mixin stacks torchvision transforms via
  `Compose`; each `_update_transform` call wraps the existing stack.
- **Distributed training** — via PyTorch Ignite's `idist` utilities
  (`auto_model`, `auto_dataloader`). Supports `DataParallel` and
  `DistributedDataParallel`.
- **Note-to-self hook** — developers leave a grep-able four-character marker
  (`xc` repeated twice) for unfinished work. A pre-commit hook blocks commits
  containing this marker.
- **Line length** — 110 characters (`ruff` enforces this).
- **Manifest files** — FITS binary tables tracking download state. These are a **known
  compromise / anti-pattern**, not a design goal. They exist because there was no better
  option at the time.
