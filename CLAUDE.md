# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hyrax is a Python-based machine learning framework for hunting rare and anomalous sources in large astronomical imaging surveys (Rubin-LSST, HSC, Euclid, NGRST). Built on PyTorch with PyTorch Ignite for training. Requires Python 3.11+.

## Essential Commands

```bash
# Development setup
bash .setup_dev.sh                        # Full setup (5-15 min)
pip install -e .'[dev]'                   # Manual alternative

# Testing
python -m pytest -m "not slow"            # Fast tests (2-5 min)
python -m pytest -m "slow"                # Slow/integration tests
python -m pytest tests/hyrax/test_file.py::test_name  # Single test

# Linting/formatting (line length: 110)
ruff check src/ tests/
ruff format src/ tests/

# Pre-commit (run before committing)
pre-commit run --all-files

# Documentation
sphinx-build -M html ./docs ./_readthedocs
```

## CLI Usage

Main entry point: `hyrax`

```bash
hyrax --help
hyrax <verb> --help
hyrax train --runtime-config config.toml  # or -c
```

**Verbs:** train, infer, download, prepare, umap, visualize, lookup, save_to_database, database_connection, test, to_onnx, model, engine, rebuild_manifest

## Architecture

```
src/
├── hyrax/
│   ├── verbs/           # CLI command implementations (inherit from Verb base class)
│   ├── models/          # PyTorch models with registry system
│   ├── data_sets/       # Dataset loaders (HSC, LSST, CIFAR, FITS, etc.)
│   ├── vector_dbs/      # ChromaDB, Qdrant implementations
│   ├── config_schemas/  # Pydantic v2 configuration validation
│   ├── hyrax.py         # Main Hyrax class (programmatic interface to verbs)
│   ├── hyrax_default_config.toml  # Default configuration
│   └── pytorch_ignite.py          # Training wrapper
└── hyrax_cli/
    └── main.py          # CLI dispatcher
```

## Key Patterns

**Verbs:** Each verb in `src/hyrax/verbs/` is a class with `setup_parser()` for CLI args and inherits from `Verb`. Registry system auto-discovers verbs.

**Models:** Defined in `src/hyrax/models/` with registry for auto-discovery. Support ONNX export. Configure via `[model]` and `[model.<ModelName>]` sections.

**Datasets:** Loaders in `src/hyrax/data_sets/` support train/validation/test splits. Configure via `[data_set]` section.

**Configuration:** TOML-based with Pydantic validation. Default config merged with runtime config passed via `-c`.

## Test Fixtures

- `loopback_hyrax` - Pre-configured Hyrax instance with random dataset
- `RandomDataset` / `RandomIterableDataset` - Test data generators

## Validation Workflow

After changes:
1. `ruff check src/ tests/ && ruff format src/ tests/`
2. `python -m pytest -m "not slow"`
3. `hyrax --help` (verify CLI works)
