# Hyrax Guide

Canonical reference for AI coding assistants working on Hyrax. Tool-specific files
(`CLAUDE.md`, `.github/copilot-instructions.md`) contain only tool-specific overrides
and reference this file for shared guidance. **Edit this file** for changes that should
apply to all AI assistants; edit tool-specific files only for tool-specific behavior.

## What Is Hyrax

Hyrax is a low-code, model-agnostic platform for machine learning on large astronomical
imaging surveys. It is built on PyTorch and PyTorch Ignite and handles the boilerplate
around downloading cutouts, building latent representations, interactive visualization,
and anomaly detection so astronomers can focus on science.

## Design Goals and North Stars
**CRITICAL: Always keep these design principles in mind when making changes to Hyrax.**

**"Configuration OR Code"** — Hyrax uses a deliberate three-tier system:

1. **Invisible** — sensible defaults handle it; the user never thinks about it.
2. **Config value** — the user sets a TOML key and Hyrax does the rest.
3. **Write code** — when the config system is not enough, the user writes a class or
   function and points the config at it via an import path.

**Jupyter notebooks are the primary interface.** The CLI (`hyrax` command) is the
secondary interface, intended for HPC / Slurm batch jobs. The CLI should be able to do
everything notebooks can. Also, design hyrax such that any dataset or model class 
(or other customization) should be authorable in a notebook, and moved to an external 
class later.

**Make Easy Things Easy, Hard Things Possible**
- **Default workflows should "just work"**: Common use cases should require minimal configuration, but in 
cases where configuration is necessary we do require it so the user is not surprised (e.g. Which ML model is 
running?)
- **Progressive complexity**: Simple tasks should be simple; advanced features available when needed
- **Sensible defaults**: Default configurations in `hyrax_default_config.toml` should handle common scenarios
- **Extensibility without complexity**: Advanced users can extend with custom models, datasets, and verbs
- **Clear extension points**: Well-documented base classes (`Verb`, model base classes, dataset classes)
- **Avoid adding Verbs**: Only add a new verb if specifically told to do so.
- **Avoid adding new configs**: Too many configs present a harder learning curve for the user. If you must 
add a config, prefer a default that works for 90% of use cases.
- **Remember our users and extenders ARE NOT Software Engineers**: Externally facing notebook interfaces and 
even user-defined (dataset, model) classes need to be written so they can be understood by hyrax users. The 
vast majority of hyrax users can write python code in a single file or notebook, but don't really understand 
classes, multi-file projects, or anything more complex. 

**Results directories are the backbone of reproducibility.** Each run creates a
timestamped directory (`YYYYMMDD-HHMMSS-<verb>-<uid>`) under `results/` containing
model weights, config snapshots, and MLflow tracking data. 
- **Configuration as documentation**: Config files serve as complete records of how experiments were run
- **Version everything**: Track model versions, data versions, and configuration versions
- **Manifest files**: Maintain manifests of downloaded data and processed results
- **Deterministic defaults**: Random seeds and other sources of variability should be configurable
- **Take your data to go**: Items in results directories should be self-contained and easy for a scientific 
user to examine outside of hyrax.
- **ONNX export**: Support model serialization for long-term reproducibility

**Target scale:** 10M–100M objects on a unix filesystem.

**Document current behavior.** When migrating away from old patterns, use clear error
messages to guide users rather than silently supporting legacy behavior. When writing documentation, prefer 
compact inspirational examples to demonstrate the breadth of the framework.

**Smooth and Legible Migration When APIs Change**
- **Clear deprecation warnings**: When changing APIs, provide helpful deprecation messages
- **Error guided Migration**: Documentation tells how the current thing works. Errors explain what 
documentation to follow to move from old to new.
- **Backward compatibility when possible**: Maintain compatibility or provide clear upgrade path

## Aspirational Goals
**Leave space for these to be implemented someday by keeping the "right now" invariants but DO NOT IMPLEMENT THE ASPIRATIONAL GOAL**

**Hyrax will someday support non-pytorch ML frameworks**
- Right now we keep all ML tensors in numpy format until the moment PyTorch needs them
- Right now all verb and dataset classes communicate in numpy format over their interfaces

**Someday there will be an ecosystem of datasets and models easily selectable by the user**
- Right now Dataset classes should work with each other a-la-carte via DataProvider
- Most dataset classes will be external libraries, but not many examples exist presently.

**Someday we will support iterable datasets**
- Right now Datasets and DataProvider always have a length and map-style access by index, and are presumed to fit in memory.
- Future iterable datasets will have Datasets and DataProvider as a finite and loadable subset of an infinite data stream

## Coding advice
When changing code, ensure that the current assumptions of the change appear to have always been true. Leave code better than 
you find it over keeping old assumptions around.

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
- Models must inherit from `torch.nn.Module` and implement `__init__`, `forward`,
  `train_step`, and `prepare_inputs`.
- The decorator wires up save/load, optimizer, and criterion handling.
- Built-in: `HyraxAutoencoder`, `HyraxAutoencoderV2`, `HyraxCNN`, `SimCLR`, `ImageDCAE`, `HSCAutoencoder`, `HSCDCAE`, `HyraxLoopback`
- **External plugins supported** — use a fully qualified import path in the config
  (e.g. `model.name = "my_pkg.my_module.MyModel"`).

### DATASET_REGISTRY (`src/hyrax/data_sets/data_set_registry.py`)
- Registration: automatic via `HyraxDataset.__init_subclass__`
- Built-in: `HyraxCifarDataset`, `HSCDataSet`, `LSSTDataset`, `FitsImageDataSet`
- **External plugins supported** — same import-path mechanism as models.

### VERB_REGISTRY (`src/hyrax/verbs/verb_registry.py`)
- Decorator: `@hyrax_verb`; base class: `Verb`
- **New verbs must be class-based**: subclass `Verb`, implement `setup_parser`, `run_cli`,
  and `run`.
- Some legacy verbs (download, prepare, rebuild_manifest) are function-based in `hyrax.py`.
  Leave these alone; do not add new function-based verbs.
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

Pydantic validation exists **only for `[data_request]`** (`config_schemas/data_request.py`)
due to that section's complexity with nested dictionaries. **Do not add Pydantic validation
to other config sections** — the rest of the config is validated by checking keys against
defaults, not by Pydantic schemas.

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

- **File naming:** `tests/hyrax/test_<name>.py`
- **Markers:** `slow` for integration / E2E tests; unmarked tests are fast.
- Default test run: `python -m pytest -m "not slow"`
- Test data: `HyraxCifarDataset` (CIFAR-10 via torchvision), `HyraxRandomDataset`,
  and Pooch/Zenodo-hosted files for slow tests.
- Parallel execution: `pytest -n auto` (pytest-xdist).
- E2E tests exercise full pipelines (train → infer → umap → visualize).

## Key Conventions

- **Spelling:** `Dataset` (lowercase 's') is preferred for new code. Legacy classes like
  `HSCDataSet` use `DataSet` — leave existing code alone, but use `Dataset` for new classes.
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
  option at the time. If extending manifest files seems like the right solution, ask the
  user for clarification first.

## Common Tasks and Workflows

### Working with Models
- Models defined in `src/hyrax/models/`
- Built-in models: `HyraxAutoencoder`, `HyraxCNN`
- Model registry system automatically discovers models
- General model configuration in `[model]` section of config files
- Configurations for specific models in `[model.<ModelName>]` sections
- Training via `hyrax train` command
- Export to ONNX format supported

### Working with Data
- Data loaders in `src/hyrax/data_sets/`
- Built-in datasets: `HSCDataSet`, `HyraxCifarDataset`, `LSSTDataset`, `FitsImageDataSet`
- Dataset splits: train/validation/test controlled by config
- Configuration in `[data_set]` section
- Default data directory: `./data/`
- Sample data includes HSC1k dataset for testing

### Working with Vector Databases
- Implementations in `src/hyrax/vector_dbs/`
- Supported: ChromaDB, Qdrant
- Commands: `save_to_database`, `database_connection`
- Configuration in `[vector_db]` section

## Notebook Development
- Jupyter integration via `holoviews`, `bokeh` for visualizations
- Interactive visualization via `hyrax visualize` verb
- Pre-executed examples in `docs/pre_executed/`

## CI/CD and GitHub Workflows
- Main workflows in `.github/workflows/`
- **Testing**: `testing-and-coverage.yml` runs on PRs and main branch
- **Smoke test**: `smoke-test.yml` runs daily
- **Documentation**: `build-documentation.yml` builds docs
- **Benchmarks**: ASV benchmarks via `asv-*.yml` workflows
- **Pre-commit**: Automated via `pre-commit-ci.yml`

## Troubleshooting
- **Import errors**: Ensure `pip install -e .'[dev]'` completed successfully
- **Network timeouts during install**: Retry installation multiple times, may require 3-5 attempts due to PyPI connectivity issues
- **ReadTimeoutError**: Common during installation - wait 1-2 minutes and retry the same pip command
- **CLI not found**: Verify installation with `pip list | grep hyrax`
- **Tests failing**: Check if in virtual environment and dependencies installed
- **Pre-commit issues**: Run `pre-commit install` if hooks not working
- **Permission issues**: Use `--user` flag with pip if encountering permission errors
- **Virtual environment**: Always use conda/venv to avoid system Python conflicts

## Performance Notes
- Vector database operations can be slow with large datasets
- Benchmarks available in `benchmarks/` directory (run with `asv` tool)
- Use `--timeout` parameters appropriately for long-running operations
- ChromaDB performance degrades with vectors >10,000 elements
- UMAP fitting limited to 1024 samples by default for performance
- Benchmark tests include timing for CLI help commands, object construction, and vector DB operations
