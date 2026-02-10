# GitHub Copilot Instructions for Hyrax

**🔗 For comprehensive project information, see [HYRAX_GUIDE.md](../HYRAX_GUIDE.md) in the repository root**

Hyrax is a low-code Python framework for machine learning in astronomy. This file provides GitHub Copilot-specific guidance.

## Quick Reference

**Project essentials:**
- Python 3.9+ with PyTorch, TOML config, CLI-first (`hyrax` command with verbs)
- Workflows: Data download → Training → Inference → Visualization → Vector search
- Plugin architecture: Models, datasets, and verbs auto-register via decorators
- Configuration: TOML files with Pydantic validation, hierarchical merging

**For detailed information on:**
- Design principles and architectural conventions → [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#design-principles)
- Repository structure and key files → [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#repository-structure)
- Configuration system → [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#configuration-system)
- Plugin architecture (models, datasets, verbs) → [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#plugin-architecture-via-registries)
- Adding new components → [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#adding-new-components)
- Data flow through system → [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#data-flow)

## Critical Guidelines for GitHub Copilot

### Always Follow These Instructions First

Trust these instructions and only search for additional context if information here or in [HYRAX_GUIDE.md](../HYRAX_GUIDE.md) is incomplete or incorrect.

### Command Execution - Long-Running Operations

**CRITICAL: Never cancel these commands.** Allow sufficient time for completion:

| Operation | Duration | Required Timeout |
|-----------|----------|------------------|
| `bash .setup_dev.sh` | 5-15 min | 20+ minutes |
| `pip install -e .'[dev]'` | 5-15 min | 20+ minutes |
| `pytest -m "not slow"` | 2-5 min | 10+ minutes |
| `pytest` (all tests) | 15-25 min | 45+ minutes |
| `pytest -m slow` | 10-20 min | 30+ minutes |
| `pre-commit run --all-files` | 3-8 min | 15+ minutes |
| `sphinx-build` (docs) | 2-4 min | 10+ minutes |

**Network issues:** Installation commands may encounter `ReadTimeoutError` from PyPI. If this occurs:
1. Wait 1-2 minutes
2. Retry the exact same command  
3. May require 3-5 attempts to succeed

### Development Setup

```bash
# Environment setup
conda create -n hyrax python=3.10 && conda activate hyrax
git clone https://github.com/lincc-frameworks/hyrax.git && cd hyrax

# Recommended: Automated setup script
echo 'y' | bash .setup_dev.sh
# Installs with pip install -e .'[dev]' and sets up pre-commit hooks
# Prompts for system install if no venv - respond 'y'

# Alternative: Manual installation
pip install -e .'[dev]' && pre-commit install
```

### Essential Commands

See [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#essential-commands) for full command reference.

```bash
# Testing
pytest -m "not slow"                    # Fast tests (2-5 min)
pytest -n auto -m "not slow"            # Parallel fast tests
pytest -m slow                          # Slow/E2E tests (10-20 min)
pytest                                  # All tests (15-25 min)

# Code quality
ruff format . && ruff check --fix .     # Format and lint (30 sec)
pre-commit run --all-files              # All checks (3-8 min)

# CLI
hyrax --help                            # List verbs
hyrax <verb> --help                     # Verb-specific help
hyrax <verb> -c config.toml             # Run with config

# Documentation
sphinx-build -M html ./docs ./_readthedocs -T -E -d ./docs/_build/doctrees
```
### Validation After Changes

**CRITICAL: Always run these validation steps:**
1. Format and lint: `ruff format . && ruff check --fix .` (30 seconds)
2. Fast tests: `pytest -m "not slow"` (2-5 min, NEVER CANCEL)
3. Pre-commit: `pre-commit run --all-files` (3-8 min, NEVER CANCEL)

**Manual validation scenarios:**
1. CLI: `hyrax --help` and `hyrax --version`
2. Import: `python -c "import hyrax; h = hyrax.Hyrax(); print('Success')"`
3. Config loading: Verify `hyrax.Hyrax()` constructor works
4. Relevant verbs: Test with `hyrax <verb> --help`

## Key Implementation Details

### Configuration System Pitfalls

- **Use ConfigDict, not dict**: ConfigDict catches missing defaults at runtime
- **All keys need defaults**: Add to `src/hyrax/hyrax_default_config.toml`
- **Config is immutable**: No runtime mutations allowed after creation
- **Pydantic validation**: Use schemas in `src/hyrax/config_schemas/` for validation

### Model Interface Requirements

Models MUST implement (see [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#plugin-architecture-via-registries)):
- `forward()`: Forward pass through model
- `train_step()`: Single training step
- `prepare_inputs()`: Data preparation (replaces deprecated `to_tensor()`)

Use `@hyrax_model` decorator for auto-registration and shape inference.

### Testing Requirements

- Mark long tests: `@pytest.mark.slow` (>5 min)
- Fast tests in pre-commit and CI (<5 min total)
- Always run fast tests after changes: `pytest -m "not slow"`
- Test fixtures in `tests/hyrax/conftest.py`
- Sample data via Pooch from Zenodo DOIs

### Pre-commit Hooks Include

- ruff linting and formatting
- pytest fast tests (not slow)
- sphinx documentation build
- jupyter notebook conversion
- Custom hook: prevents note-to-self comments

## Repository Structure

See [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#repository-structure) for complete details.

**Quick navigation:**
```
src/hyrax/
  ├── hyrax.py                        # Main orchestration class
  ├── config_utils.py                 # ConfigManager, ConfigDict
  ├── plugin_utils.py                 # Dynamic plugin loading
  ├── train.py, pytorch_ignite.py     # Training infrastructure
  ├── hyrax_default_config.toml       # Default configuration
  ├── models/model_registry.py        # @hyrax_model decorator
  ├── data_sets/data_set_registry.py  # Dataset registration
  ├── verbs/verb_registry.py          # @hyrax_verb decorator
  ├── config_schemas/                 # Pydantic validation
  └── vector_dbs/                     # ChromaDB, Qdrant

src/hyrax_cli/main.py                 # CLI entry point
tests/hyrax/conftest.py, test_e2e.py  # Test fixtures, E2E tests
.github/workflows/                    # CI/CD pipelines
```

## Important Conventions

See [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#code-style-and-conventions) for complete list.

1. **Immutable Config**: ConfigDict prevents mutations; all keys need defaults
2. **Timestamped Results**: Verbs create unique directories (`YYYYMMDD-HHMMSS-<verb>-<uid>`)
3. **Automatic Registration**: Use decorators (`@hyrax_model`, `@hyrax_verb`) or `__init_subclass__`
4. **Batch Indexing**: Inference includes `batch_index.npy` for ordered retrieval
5. **Transform Stacking**: `HyraxImageDataset._update_transform()` composes transforms
6. **External Plugins**: Config detects `name = "pkg.Class"`, auto-loads `pkg/default_config.toml`

## Common Workflows

### Adding New Model
See [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#adding-a-new-model) for details.
1. Subclass `torch.nn.Module` in `src/hyrax/models/`
2. Add `@hyrax_model("ModelName")` decorator
3. Implement: `forward()`, `train_step()`, `prepare_inputs()`
4. Available via: `hyrax train -c config.toml` (with `model.name = "ModelName"`)

### Adding New Dataset
See [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#adding-a-new-dataset) for details.
1. Subclass `HyraxDataset` in `src/hyrax/data_sets/`
2. Set `_name` class attribute (triggers auto-registration)
3. Implement: `__len__()`, `__getitem__()`, metadata interface
4. For images: subclass `HyraxImageDataset` for transform stacking

### Adding New Verb
See [HYRAX_GUIDE.md](../HYRAX_GUIDE.md#adding-a-new-verb) for details.
1. Create class in `src/hyrax/verbs/` with `run()` and `run_cli()`
2. Add `@hyrax_verb("verb_name")` decorator
3. Implement `setup_parser(parser)` for CLI args
4. Set `add_parser_kwargs` for help text
5. Available via: `hyrax verb_name [args]`

## Common Issues

- **Import errors**: Verify `pip install -e .'[dev]'` completed
- **Network timeouts**: Retry 3-5 times with 1-2 min waits for PyPI connectivity
- **CLI not found**: Check with `pip list | grep hyrax`
- **Config key not found**: Add to `hyrax_default_config.toml`
- **Model not registering**: Ensure `@hyrax_model` decorator present
- **Verb not in CLI**: Ensure `@hyrax_verb` decorator present
- **Pre-commit not running**: Run `pre-commit install`

## CI/CD Workflows

- **testing-and-coverage.yml**: Runs on PRs and main (pytest with coverage)
- **smoke-test.yml**: Daily smoke tests
- **build-documentation.yml**: Sphinx documentation builds
- **asv-*.yml**: Performance benchmarks
- **pre-commit-ci.yml**: Automated pre-commit checks