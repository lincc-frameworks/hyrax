# Hyrax Development Guide

This guide provides essential information for working with the Hyrax codebase. It is referenced by both CLAUDE.md and .github/copilot-instructions.md.

## Project Overview

Hyrax is a Python-based tool for hunting rare and anomalous sources in large astronomical imaging surveys. It provides a low-code solution for rapid experimentation with machine learning in astronomy.

### Core Purpose
Hyrax helps scientists/astronomers handle much of the boilerplate code that is often required for a machine learning project in astronomy so that users can focus on their model development and downstream science.

### Primary Workflows
1. **Data Access**: Downloading/accessing data from specific public data repositories (e.g. HSC, Rubin-LSST)
2. **Training**: Training supervised/unsupervised ML algorithms using astronomical data
3. **Inference**: Performing inference to generate latent representations
4. **Visualization**: Building interactive 2D and 3D latent spaces
5. **Vector Search**: Building vector databases with inference results for rapid similarity search and outlier detection

### Technology Stack
- **Language**: Python >= 3.9 (target 3.9 for compatibility)
- **ML Framework**: PyTorch with PyTorch Ignite for distributed training
- **Configuration**: TOML-based hierarchical configuration system
- **CLI**: Verb-based command interface via `hyrax` command
- **Testing**: pytest with parallel execution support
- **Linting**: ruff (replaces black, isort, flake8)
- **Documentation**: Sphinx with ReadTheDocs

## Design Principles

### 1. Low Code Interface
- Minimize user-facing APIs - prioritize configuration-driven workflows
- Avoid API proliferation - don't create new APIs we'll need to maintain indefinitely
- Favor declarative over imperative configuration
- CLI-first approach with verb-based commands (`hyrax train`, `hyrax infer`, etc.)

### 2. Make Easy Things Easy, Hard Things Possible
- Default workflows should "just work" with minimal configuration
- Progressive complexity - simple tasks simple, advanced features available when needed
- Sensible defaults in `hyrax_default_config.toml`
- Clear extension points via base classes (`Verb`, model base classes, dataset classes)

### 3. Support Reproducibility
- Configuration files serve as complete records of experiments
- Version tracking for models, data, and configurations
- Manifest files for downloaded data and processed results
- MLflow integration for systematic experiment logging
- ONNX export support for long-term reproducibility

### 4. Smooth and Legible Migration When APIs Change
- Clear deprecation warnings with helpful messages
- Migration guides in documentation with before/after examples
- Backward compatibility when possible
- Pydantic schemas for config validation with helpful error messages
- Comprehensive changelog with breaking change notifications

## Development Setup

### Environment Setup
```bash
# Create and activate virtual environment
conda create -n hyrax python=3.10
conda activate hyrax

# Clone repository
git clone https://github.com/lincc-frameworks/hyrax.git
cd hyrax

# Install for development (recommended)
bash .setup_dev.sh
# This script:
# - Installs package with pip install -e .'[dev]'
# - Sets up pre-commit hooks
# - Takes 5-15 minutes depending on network
# - Prompts for system install if no venv detected - respond 'y'

# Alternative manual installation
pip install -e .'[dev]'           # Install with dev dependencies
pre-commit install                 # Set up pre-commit hooks
```

### Common Issues During Setup
- **ReadTimeoutError**: Installation may fail due to PyPI connectivity - retry multiple times if needed
- **Permission errors**: Use `--user` flag with pip if encountering permission errors
- **Virtual environment**: Always use conda/venv to avoid system Python conflicts

## Essential Commands

### Testing
```bash
# Fast tests (default, excludes slow tests)
pytest -m "not slow"                                    # 2-5 minutes
pytest -n auto -m "not slow"                            # Parallel execution

# Tests with coverage
pytest -n auto --cov=./src --cov-report=html -m "not slow"

# Slow tests (includes end-to-end tests)
pytest -m slow                                          # 10-20 minutes

# All tests
pytest                                                  # 15-25 minutes
pytest -n auto                                          # Parallel, faster

# Specific test file or function
pytest tests/hyrax/test_config_utils.py
pytest tests/hyrax/test_infer.py::test_infer_basic
```

### Code Quality
```bash
# Linting and formatting
ruff check --fix .                                      # 10-30 seconds
ruff format .                                           # 10-30 seconds

# Pre-commit hooks (run all checks)
pre-commit run --all-files                              # 3-8 minutes

# Build documentation
sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees
```

### CLI Usage
```bash
# Get help
hyrax --help                        # List all verbs/commands
hyrax --version                     # Show version
hyrax <verb> --help                 # Help for specific verb

# Common verbs
hyrax train -c config.toml          # Train a model
hyrax infer -c config.toml          # Generate latent representations
hyrax umap -c config.toml           # Dimensionality reduction
hyrax visualize                     # Interactive visualization
hyrax save_to_database              # Populate vector DB
hyrax lookup                        # Query vector DB
```

## Architecture Overview

### Plugin Architecture via Registries

Hyrax uses three primary registries for extensibility:

1. **MODEL_REGISTRY** (`models/model_registry.py`)
   - Maps model names to PyTorch nn.Module classes
   - `@hyrax_model` decorator auto-registers models
   - Models must implement: `forward()`, `train_step()`, `prepare_inputs()` (formerly `to_tensor()`)
   - Automatic shape inference from dataset samples

2. **DATA_SET_REGISTRY** (`data_sets/data_set_registry.py`)
   - Maps dataset names to HyraxDataset classes
   - Auto-registration via `__init_subclass__` when subclasses defined
   - Base class provides metadata interface, ID generation, catalog access

3. **VERB_REGISTRY** (`verbs/verb_registry.py`)
   - Maps CLI command names to Verb classes
   - `@hyrax_verb` decorator registers verbs
   - Verbs can be class-based (`run()` and `run_cli()` methods) or function-based

### Configuration System

- **TOML-based hierarchical configuration** with strong validation via Pydantic schemas
- **ConfigManager** merges: `hyrax_default_config.toml` + external library configs + user runtime config
- **ConfigDict** enforces all keys must have defaults (prevents silent config bugs)
- Automatic path resolution for relative paths
- Config sections: `[general]`, `[model]`, `[train]`, `[data_set]`, `[download]`, etc.

### External Plugin Support

External libraries can provide custom models/datasets/verbs:
1. Set config values like `name = "external_pkg.model.CustomModel"`
2. Provide a `default_config.toml` file in the package root
3. Hyrax's `get_or_load_class()` in `plugin_utils.py` handles dynamic import and config merging

### Data Flow

```
DOWNLOAD (optional)
  ↓ Catalog (FITS) → Downloader → Cutout images + manifest.fits
  
PREPROCESSING (implicit in dataset)
  ↓ Dataset loads raw images → applies transforms → train/validate/test splits
  
TRAINING
  ↓ train.py: setup_dataset → setup_model → create_trainer → checkpoints + MLflow logs
  
INFERENCE
  ↓ Model.forward(batch) → latent vectors → batch_*.npy files + batch_index.npy
  
VECTOR DB / VISUALIZATION
  ↓ ChromaDB for similarity search | UMAP → 2D/3D → Holoviews scatter plot
```

### Key Abstractions

**Hyrax class** (`hyrax.py`): Central orchestration interface wrapping all functionality. Provides both programmatic and CLI access via dynamic `__getattr__` that instantiates verb classes on demand.

**HyraxDataset** (`data_sets/`): Base class for all datasets
- Subclasses auto-register via `__init_subclass__`
- Must provide metadata interface (fields, catalog data)
- `HyraxImageDataset` mixin provides transform stacking via `_update_transform()`
- Built-in: HSCDataSet, LSSTDataset, FitsImageDataSet, HyraxCifarDataSet, InferenceDataSet

**Model Registration**: `@hyrax_model` decorator provides:
- Automatic shape inference by sampling dataset
- Standardized save/load via PyTorch state_dict
- Criterion and optimizer loading from config

**Verb Pattern**: Base `Verb` class with `run()` (programmatic) and `run_cli()` (CLI) methods
- CLI autodiscovery via `all_verbs()` in registry
- Class-based: Infer, Umap, Visualize, SaveToDatabase, Lookup
- Function-based: train, download, prepare, rebuild_manifest

**Result Chaining**: Verbs create timestamped directories (`YYYYMMDD-HHMMSS-<verb>-<uid>`)
- `find_most_recent_results_dir()` enables automatic chaining between verbs
- InferenceDataSet preserves original dataset config for metadata access

### Training Infrastructure

- **PyTorch Ignite-based** distributed training (`pytorch_ignite.py`, `train.py`)
- `setup_dataset()`: Instantiates dataset from config
- `setup_model()`: Instantiates model, infers shape from dataset
- `dist_data_loader()`: Creates distributed data loaders with splits
- `create_trainer()`: Training engine with checkpointing, progress bars
- MLflow for experiment tracking, TensorboardX for metric logging

## Repository Structure

### Key Directories
```
src/hyrax/                  # Main package source code
  ├── models/               # Model definitions
  ├── data_sets/            # Dataset implementations
  ├── verbs/                # Command implementations
  ├── vector_dbs/           # Vector database implementations (ChromaDB, Qdrant)
  └── config_schemas/       # Pydantic schemas for configuration validation
src/hyrax_cli/              # CLI entry point (main.py)
tests/hyrax/                # Unit and integration tests
docs/                       # Documentation source files
benchmarks/                 # Performance benchmarks (ASV)
example_notebooks/          # Example Jupyter notebooks
```

### Important Files
- `pyproject.toml`: Project configuration, dependencies, CLI entry points
- `src/hyrax/hyrax_default_config.toml`: Default configuration template
- `.setup_dev.sh`: Development environment setup script
- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `.github/workflows/`: CI/CD pipeline definitions

## Code Style and Conventions

- **Line length**: 110 characters (configured in pyproject.toml)
- **Docstrings**: Required for public classes and functions (enforced by ruff D101-D106)
- **Pre-commit hooks**: Automatically run on commit (ruff, pytest, sphinx-build, jupyter nbconvert)
- **No note-to-self comments**: Custom pre-commit hook prevents placeholder comments

### Important Architectural Conventions

1. **Immutable Config**: ConfigDict prevents runtime mutations; all keys must have defaults
2. **Timestamped Results**: Every verb execution creates unique directory preventing overwrites
3. **Metadata Preservation**: InferenceDataSet stores original dataset config to maintain catalog access
4. **Automatic Registration**: Use decorators (`@hyrax_model`, `@hyrax_verb`) or `__init_subclass__` - no manual registration
5. **Batch Indexing**: Inference results include `batch_index.npy` mapping object_ids → batch files
6. **Transform Stacking**: HyraxImageDataset uses `_update_transform()` to compose torchvision transforms
7. **Distributed Training**: PyTorch Ignite's `idist.auto_dataloader()` abstracts single/multi-GPU execution
8. **External Library Support**: Config system detects `name = "pkg.Class"` and auto-loads `pkg/default_config.toml`

## Testing Conventions

- **End-to-end tests** in `test_e2e.py` parametrized across model/dataset combinations
- **Test markers**: `@pytest.mark.slow` for long-running tests (skipped in pre-commit and CI)
- **Test fixtures** in `tests/hyrax/conftest.py` provide shared setup
- **Sample data**: Uses Pooch for reproducible downloads from Zenodo DOIs
- **Pre-commit**: Runs fast tests only: `pytest -n auto --cov=./src -m 'not slow'`

## CI/CD

- **Testing**: `testing-and-coverage.yml` runs on PRs and main branch
- **Smoke test**: `smoke-test.yml` runs daily
- **Documentation**: `build-documentation.yml` builds docs
- **Benchmarks**: ASV benchmarks via `asv-*.yml` workflows
- **Pre-commit**: Automated via `pre-commit-ci.yml`

## Adding New Components

### Adding a New Model
1. Subclass `torch.nn.Module` in `src/hyrax/models/`
2. Add `@hyrax_model` decorator with unique name
3. Implement: `forward()`, `train_step()`, `prepare_inputs()`
4. Available via CLI: `hyrax train -c config.toml` (with `model.name = "YourModelName"`)

### Adding a New Dataset
1. Subclass `HyraxDataset` in `src/hyrax/data_sets/`
2. Set `_name` class attribute (triggers auto-registration)
3. Implement: `__len__()`, `__getitem__()`, metadata interface
4. For images, subclass `HyraxImageDataset` to get transform stacking

### Adding a New Verb
1. Create class in `src/hyrax/verbs/` with `run()` and optionally `run_cli()`
2. Add `@hyrax_verb("verb_name")` decorator
3. Implement `setup_parser(parser)` class method for CLI argument parsing
4. Set `add_parser_kwargs` class attribute for help text
5. Available via CLI: `hyrax verb_name [args]`

## Troubleshooting

- **Import errors**: Ensure `pip install -e .'[dev]'` completed successfully
- **Network timeouts**: Retry installation multiple times (3-5 attempts may be needed)
- **CLI not found**: Verify with `pip list | grep hyrax`
- **Tests failing**: Check virtual environment and dependencies
- **Pre-commit issues**: Run `pre-commit install` if hooks not working

## Performance Notes

- Vector database operations can be slow with large datasets
- ChromaDB performance degrades with vectors >10,000 elements
- UMAP fitting limited to 1024 samples by default for performance
- Benchmarks available in `benchmarks/` directory (run with `asv` tool)
