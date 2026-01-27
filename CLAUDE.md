# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Hyrax is designed to be a low-code solution for rapid experimentation with machine learning in astronomy

Hyrax helps scientists/astronomers handle much of the boilerplate code that is often required for a machine learning project in astronomy so that users can focus on their model development and downstream science. 

Hyrax supports a few primary workflows:
1. Downloading/Accessing data from specific public data repositories (e.g. HSC, Rubin-LSST)
2. Training supervised/unsupervised ML algorithms using the above data or other data a user chooses to bring to Hyrax
3. Performing inference using the above models
4. Building interactive two and three dimensional latent spaces using the above tools 
4. Building vector databased with inference results for rapid similarity search and outlier detection. 

Hyrax is model-agnostic and extensible, supporting any PyTorch-based algorithm.

## Development Setup

```bash
# Clone and setup environment
git clone https://github.com/lincc-frameworks/hyrax.git
conda create -n hyrax python=3.10
conda activate hyrax

# For developers - installs package in editable mode, dev dependencies, and pre-commit hooks
bash .setup_dev.sh

# Manual installation
pip install -e .              # Runtime dependencies only
pip install -e .'[dev]'       # Include dev dependencies
```

## Common Commands

### Testing
```bash
# Run all tests (excluding slow tests)
pytest -m "not slow"

# Run tests in parallel
pytest -n auto -m "not slow"

# Run with coverage
pytest -n auto --cov=./src --cov-report=html -m "not slow"

# Run slow tests (includes end-to-end tests)
pytest -m slow

# Run specific test file
pytest tests/hyrax/test_config_utils.py

# Run specific test function
pytest tests/hyrax/test_infer.py::test_infer_basic
```

### Linting
```bash
# Run ruff linting with auto-fix
ruff check --fix .

# Format code with ruff
ruff format .

# Run pre-commit hooks manually
pre-commit run --all-files
```

### Documentation
```bash
# Build documentation locally
sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees
```

### CLI Usage
```bash
# Show available verbs (commands)
hyrax --help

# Show version
hyrax --version

# Run with custom config
hyrax <verb> -c path/to/config.toml

# Common verbs
hyrax train           # Train a model
hyrax infer           # Run inference to generate latent space
hyrax umap            # Dimensionality reduction on inference results
hyrax visualize       # Interactive visualization
hyrax save_to_database  # Populate vector DB from inference
hyrax lookup          # Query vector DB
```

## Architecture Overview

### Core Design Pattern: Plugin Architecture via Registries

Hyrax uses three primary registries for extensibility:

1. **MODEL_REGISTRY** (`models/model_registry.py`): Maps model names to PyTorch nn.Module classes
   - The `@hyrax_model` decorator auto-registers models and injects standard interface methods
   - Models must implement: `forward()`, `train_step()`, `to_tensor()`
   - Automatic shape inference from dataset samples

2. **DATA_SET_REGISTRY** (`data_sets/data_set_registry.py`): Maps dataset names to HyraxDataset classes
   - Uses `__init_subclass__` for automatic registration when subclasses are defined
   - Base class provides metadata interface, ID generation, catalog access

3. **VERB_REGISTRY** (`verbs/verb_registry.py`): Maps CLI command names to Verb classes
   - The `@hyrax_verb` decorator registers verbs
   - Verbs can be class-based (with `run()` and `run_cli()` methods) or function-based

### External Plugin Support

External libraries can provide custom models/datasets/verbs by:
1. Setting config values like `name = "external_pkg.model.CustomModel"`
2. Providing a `default_config.toml` file in the package root
3. Hyrax's `get_or_load_class()` in `plugin_utils.py` handles dynamic import and config merging

### Configuration System

- **TOML-based hierarchical configuration** with strong validation
- **ConfigManager** merges: `hyrax_default_config.toml` + external library configs + user runtime config
- **ConfigDict** enforces all keys must have defaults (prevents silent config bugs)
- Automatic path resolution for relative paths
- Use `ConfigDict` instead of regular dict in new code to catch missing defaults at runtime

### Data Flow Through the System

```
1. DOWNLOAD (optional)
   - Catalog (FITS) → Downloader → Cutout images + manifest.fits
   - Stored in config[general][data_dir]

2. PREPROCESSING (implicit in dataset)
   - Dataset loads raw images → applies transforms (crop, tanh, etc.)
   - Split into train/validate/test via SubsetSequentialSampler
   - DataLoader batching with optional caching

3. TRAINING
   - train.py orchestrates: setup_dataset → setup_model → create_trainer
   - Model.train_step() called per batch
   - Checkpoints saved to timestamped results_dir
   - MLflow logs metrics/params

4. LATENT SPACE (Inference)
   - Infer verb: Model.forward(batch) → latent vectors
   - InferenceDataSetWriter saves: batch_<N>.npy files + batch_index.npy
   - Optional: SaveToDatabase → ChromaDB for similarity search
   - Umap verb: reduces latent space to 2D/3D

5. VISUALIZATION/SEARCH
   - Visualize: InferenceDataSet reads umap results → Holoviews scatter plot
   - Lookup: Query ChromaDB by ID or vector → k-nearest neighbors
```

### Key Abstractions

**Hyrax class** (`hyrax.py`): Central orchestration interface that wraps all functionality. Provides both programmatic and CLI access to all verbs via dynamic `__getattr__` that instantiates verb classes on demand.

**HyraxDataset** (`data_sets/`): Base class for all datasets
- Subclasses automatically register via `__init_subclass__`
- Must provide metadata interface (fields, catalog data)
- `HyraxImageDataset` mixin provides transform stacking via `_update_transform()`
- Built-in datasets: HSCDataSet, LSSTDataset, FitsImageDataSet, HyraxCifarDataSet, InferenceDataSet

**Model Registration**: The `@hyrax_model` decorator provides:
- Automatic shape inference by sampling the dataset
- Standardized save/load via PyTorch state_dict
- Criterion and optimizer loading from config
- Injection of common interface methods

**Verb Pattern**: Base `Verb` class with `run()` (programmatic) and `run_cli()` (CLI) methods
- CLI autodiscovery via `all_verbs()` in registry
- Class-based verbs: Infer, Umap, Visualize, SaveToDatabase, Lookup, DatabaseConnection
- Function-based verbs: train, download, prepare, rebuild_manifest

**Result Chaining**: Verbs create timestamped result directories (`YYYYMMDD-HHMMSS-<verb>-<uid>`)
- `find_most_recent_results_dir()` enables automatic chaining between verbs
- InferenceDataSet preserves original dataset config for metadata access

### Training Infrastructure

- **PyTorch Ignite-based** distributed training (`pytorch_ignite.py`, `train.py`)
- `setup_dataset()`: Instantiates dataset from config
- `setup_model()`: Instantiates model, infers shape from dataset
- `dist_data_loader()`: Creates distributed data loaders with train/validate/test splits
- `create_trainer()`: Training engine with checkpointing, progress bars
- MLflow integration for experiment tracking
- TensorboardX for metric logging

### Testing Conventions

- **End-to-end tests** in `test_e2e.py` are parametrized across model/dataset combinations
- Use `@pytest.mark.slow` for long-running tests (skipped in pre-commit and CI)
- Test fixtures in `tests/hyrax/conftest.py` provide shared setup
- Sample data uses Pooch for reproducible downloads from Zenodo DOIs
- Pre-commit hook runs fast tests only: `pytest -n auto --cov=./src -m 'not slow'`

## Important Architectural Conventions

1. **Immutable Config**: ConfigDict prevents runtime mutations; all keys must have defaults
2. **Timestamped Results**: Every verb execution creates a unique directory preventing overwrites
3. **Metadata Preservation**: InferenceDataSet stores original dataset config to maintain catalog access
4. **Automatic Registration**: Use decorators (`@hyrax_model`, `@hyrax_verb`) or `__init_subclass__` - no manual registration
5. **Batch Indexing**: Inference results include `batch_index.npy` mapping object_ids → batch files (critical for ordered retrieval)
6. **Transform Stacking**: HyraxImageDataset uses `_update_transform()` to compose torchvision transforms in sequence
7. **Distributed Training**: PyTorch Ignite's `idist.auto_dataloader()` abstracts single/multi-GPU execution
8. **External Library Support**: Config system detects `name = "pkg.Class"` and auto-loads `pkg/default_config.toml`

## Code Style

- **Line length**: 110 characters (configured in pyproject.toml)
- **Python version**: >= 3.9, target 3.9 for compatibility
- **Linter**: ruff (replaces black, isort, flake8)
- **Docstrings**: Required for public classes and functions (enforced by ruff D101, D102, D103, D106)
- **Pre-commit hooks**: Run automatically on commit (includes ruff, pytest, sphinx-build, jupyter nbconvert)
- **No note-to-self comments**: Custom pre-commit hook prevents placeholder comments from being committed

## Key Files and Modules

- `src/hyrax/hyrax.py`: Main Hyrax orchestration class
- `src/hyrax/config_utils.py`: Configuration system (ConfigManager, ConfigDict)
- `src/hyrax/plugin_utils.py`: Dynamic plugin loading (`get_or_load_class`)
- `src/hyrax/train.py`: Training orchestration with PyTorch Ignite
- `src/hyrax/pytorch_ignite.py`: Setup functions for datasets, models, data loaders
- `src/hyrax_cli/main.py`: CLI entry point with auto-discovered verb subparsers
- `src/hyrax/models/model_registry.py`: Model registration and `@hyrax_model` decorator
- `src/hyrax/data_sets/data_set_registry.py`: Dataset registration
- `src/hyrax/verbs/verb_registry.py`: Verb registration and `@hyrax_verb` decorator
- `src/hyrax/vector_dbs/`: Vector database abstraction (ChromaDB implementation)
- `tests/hyrax/test_e2e.py`: End-to-end integration tests

## Adding New Components

### Adding a New Model
1. Subclass `torch.nn.Module` in `src/hyrax/models/`
2. Add `@hyrax_model` decorator with a unique name
3. Implement required methods: `forward()`, `train_step()`, `to_tensor()`
4. Model will auto-register and be available via CLI: `hyrax train -c config.toml` (with `model.name = "YourModelName"`)

### Adding a New Dataset
1. Subclass `HyraxDataset` in `src/hyrax/data_sets/`
2. Set `_name` class attribute (triggers auto-registration via `__init_subclass__`)
3. Implement required methods: `__len__()`, `__getitem__()`, metadata interface
4. For image datasets, subclass `HyraxImageDataset` to get transform stacking

### Adding a New Verb
1. Create a class in `src/hyrax/verbs/` that implements `run()` and optionally `run_cli()`
2. Add `@hyrax_verb("verb_name")` decorator
3. Implement `setup_parser(parser)` class method for CLI argument parsing
4. Set `add_parser_kwargs` class attribute for help text
5. Verb will be available via CLI: `hyrax verb_name [args]` and programmatically: `hyrax_instance.verb_name()`
