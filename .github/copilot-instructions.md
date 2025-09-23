# Hyrax - An extensible Framework for Machine Learning in Astronomy

**ALWAYS follow these instructions first and only fallback to additional search and context gathering if the information here is incomplete or found to be in error.**

Hyrax is a Python-based tool for hunting rare and anomalous sources in large astronomical imaging surveys. It supports downloading cutouts, building latent representations, interactive visualization, and anomaly detection using PyTorch models.

## Working Effectively

### Bootstrap and Setup - NEVER CANCEL these commands
- Create virtual environment: `conda create -n hyrax python=3.10 && conda activate hyrax`
- Clone repository: `git clone https://github.com/lincc-frameworks/hyrax.git`
- **CRITICAL**: Install dependencies using `.setup_dev.sh` script:
  - `cd hyrax && echo 'y' | bash .setup_dev.sh` -- NEVER CANCEL: Takes 5-15 minutes depending on network. Set timeout to 20+ minutes.
  - Script installs package with `pip install -e .'[dev]'` and sets up pre-commit hooks
  - **Note**: Script prompts for system install if no virtual environment detected - respond 'y' to proceed
  - **Alternative manual installation** if script fails due to network issues:
    - `python -m pip install --upgrade pip` first
    - `python -m pip install -e .'[dev]'` -- NEVER CANCEL: Takes 5-15 minutes. Set timeout to 20+ minutes.
    - `python -m pip install pre-commit && pre-commit install` 
    - `conda install pandoc` (for documentation)
  - **Network Issues**: Installation may fail with ReadTimeoutError due to PyPI connectivity. Retry installation multiple times if needed.

### Build and Test Commands - NEVER CANCEL these commands
- **Run tests**: `python -m pytest -m "not slow"` -- NEVER CANCEL: Takes 2-5 minutes. Set timeout to 10+ minutes.
- **Run tests with coverage**: `python -m pytest --cov=hyrax --cov-report=xml -m "not slow"` -- NEVER CANCEL: Takes 3-6 minutes. Set timeout to 10+ minutes.
- **Run slow tests**: `python -m pytest -m "slow"` -- NEVER CANCEL: Takes 10-20 minutes. Set timeout to 30+ minutes.
- **Run all tests**: `python -m pytest` -- NEVER CANCEL: Takes 15-25 minutes. Set timeout to 45+ minutes.
- **Run parallel tests**: `python -m pytest -n auto` (uses multiple cores)

### CLI Usage and Functionality
- **Main CLI entry point**: `hyrax` command (defined in pyproject.toml as `hyrax = "hyrax_cli.main:main"`)
- **Check version**: `hyrax --version`
- **Get help**: `hyrax --help`
- **Available verbs/commands**: 
  - **Core operations**: `train`, `infer`, `download`, `prepare`
  - **Analysis**: `umap`, `visualize`, `lookup`
  - **Vector DB**: `save_to_database`, `database_connection`
  - **Utilities**: `rebuild_manifest`
- **Verb-specific help**: `hyrax <verb> --help` (e.g., `hyrax train --help`)
- **Configuration**: Use `--runtime-config path/to/config.toml` or `-c path/to/config.toml`
- **Verb implementation**: All verbs are classes in `src/hyrax/verbs/` that inherit from `Verb` base class

### Development and Code Quality - NEVER CANCEL these commands
- **Pre-commit checks**: `pre-commit run --all-files` -- NEVER CANCEL: Takes 3-8 minutes. Set timeout to 15+ minutes.
- **Linting with ruff**: `ruff check src/ tests/` -- Takes 10-30 seconds.
- **Format with ruff**: `ruff format src/ tests/` -- Takes 10-30 seconds.
- **Build documentation**: `sphinx-build -M html ./docs ./_readthedocs` -- NEVER CANCEL: Takes 2-4 minutes. Set timeout to 10+ minutes.

## Validation and Testing

### CRITICAL: Always run these validation steps after making changes
1. **NEVER CANCEL**: Lint and format code: `ruff check src/ tests/ && ruff format src/ tests/`
2. **NEVER CANCEL**: Run unit tests: `python -m pytest -m "not slow"` (timeout: 10+ minutes)
3. **NEVER CANCEL**: Run pre-commit hooks: `pre-commit run --all-files` (timeout: 15+ minutes)

### Manual Validation Scenarios
After making changes, ALWAYS test these scenarios:
1. **CLI functionality**: Run `hyrax --help` and `hyrax --version` to ensure CLI works
2. **Import test**: `python -c "import hyrax; h = hyrax.Hyrax(); print('Success')"` 
3. **Configuration loading**: Verify config loads with `hyrax.Hyrax()` constructor
4. **Verb functionality**: Test relevant verbs like `hyrax train --help` if modifying training code

### Test Categories and Markers
- **Fast tests**: `python -m pytest -m "not slow"` (default test suite)
- **Slow tests**: `python -m pytest -m "slow"` (integration and E2E tests)
- **E2E tests**: Full end-to-end workflows testing models and datasets
- **Test datasets**: Uses built-in datasets like `HyraxCifarDataSet`, `HSCDataSet`
- **Test models**: Primarily tests `HyraxAutoencoder` model
- **Parallel testing**: Use `-n auto` for multiprocessing

### Timeout Values and Timing Expectations
- **NEVER CANCEL**: Package installation: 5-15 minutes (timeout: 20+ minutes)
- **NEVER CANCEL**: Unit tests: 2-5 minutes (timeout: 10+ minutes) 
- **NEVER CANCEL**: Full test suite: 15-25 minutes (timeout: 45+ minutes)
- **NEVER CANCEL**: Pre-commit hooks: 3-8 minutes (timeout: 15+ minutes)
- **NEVER CANCEL**: Documentation build: 2-4 minutes (timeout: 10+ minutes)
- Code formatting/linting: 10-30 seconds

### Network and Installation Issues
- **PyPI Connectivity**: May encounter ReadTimeoutError when installing packages
- **Retry Strategy**: If installation fails, wait 1-2 minutes and retry the same command
- **Alternative mirrors**: Consider using `--index-url` with alternative PyPI mirrors if persistent issues
- **Dependency conflicts**: The package has complex ML dependencies (PyTorch, etc.) which may cause conflicts

## Repository Structure and Navigation

### Key Directories
- `src/hyrax/`: Main package source code
- `src/hyrax_cli/`: CLI entry point (`main.py`)
- `src/hyrax/verbs/`: Command implementations (train, infer, download, etc.)
- `src/hyrax/data_sets/`: Dataset implementations 
- `src/hyrax/models/`: Model definitions
- `src/hyrax/vector_dbs/`: Vector database implementations (ChromaDB, Qdrant)
- `tests/hyrax/`: Unit tests
- `docs/`: Documentation source files
- `benchmarks/`: Performance benchmarks
- `example_notebooks/`: Example Jupyter notebooks

### Important Files
- `pyproject.toml`: Project configuration, dependencies, scripts
- `src/hyrax/hyrax_default_config.toml`: Default configuration template
- `.setup_dev.sh`: Development environment setup script
- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `.github/workflows/`: CI/CD pipeline definitions

### Configuration System
- Default config: `src/hyrax/hyrax_default_config.toml`
- Users can override with custom config files via `--runtime-config`
- Config sections: `[general]`, `[model]`, `[train]`, `[data_set]`, `[download]`, etc.

## Common Tasks and Workflows

### Adding New Features
1. **ALWAYS** run full validation first: `python -m pytest -m "not slow"`
2. Make changes in appropriate `src/hyrax/` subdirectory
3. Add tests in `tests/hyrax/` following existing patterns
4. **ALWAYS** run: `ruff format src/ tests/ && ruff check src/ tests/`
5. **ALWAYS** run: `python -m pytest -m "not slow"` (timeout: 10+ minutes)
6. **ALWAYS** run: `pre-commit run --all-files` (timeout: 15+ minutes)

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
- Built-in datasets: `HSCDataSet`, `HyraxCifarDataSet`, `LSSTDataset`, `FitsImageDataSet`
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

## Common Command Reference
```bash
# Full development setup
conda create -n hyrax python=3.10 && conda activate hyrax
git clone https://github.com/lincc-frameworks/hyrax.git && cd hyrax
echo 'y' | bash .setup_dev.sh

# Quick validation workflow  
ruff check src/ tests/ && ruff format src/ tests/
python -m pytest -m "not slow"
pre-commit run --all-files
```