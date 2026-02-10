# CLAUDE.md

This file provides Claude Code (claude.ai/code) specific guidance when working with this repository.

**🔗 For comprehensive project information, architecture, and workflows, see [HYRAX_GUIDE.md](./HYRAX_GUIDE.md)**

## Quick Reference

Hyrax is a low-code Python tool for machine learning in astronomy. Key facts:
- **Tech Stack**: Python 3.9+, PyTorch, TOML configuration, CLI-first design
- **Main Workflows**: Data download → Training → Inference → Visualization → Vector search
- **Entry Point**: `hyrax` CLI with verb-based commands (train, infer, umap, visualize, etc.)
- **Configuration**: TOML files with Pydantic validation, hierarchical merging
- **Testing**: pytest with `@pytest.mark.slow` for long tests, parallel execution with `-n auto`

**Always refer to [HYRAX_GUIDE.md](./HYRAX_GUIDE.md) for:**
- Design principles and architectural conventions
- Repository structure and key files
- Configuration system details
- Plugin architecture (models, datasets, verbs)
- Adding new components
- Data flow through the system

## Claude-Specific Guidance

### Command Execution Strategy

**CRITICAL: Never cancel long-running commands.** Hyrax has several operations that require extended execution time:

| Command | Typical Duration | Minimum Timeout |
|---------|------------------|-----------------|
| `bash .setup_dev.sh` | 5-15 minutes | 20 minutes |
| `pip install -e .'[dev]'` | 5-15 minutes | 20 minutes |
| `pytest -m "not slow"` | 2-5 minutes | 10 minutes |
| `pytest` (all tests) | 15-25 minutes | 45 minutes |
| `pytest -m slow` | 10-20 minutes | 30 minutes |
| `pre-commit run --all-files` | 3-8 minutes | 15 minutes |
| `sphinx-build ...` | 2-4 minutes | 10 minutes |
| `ruff check/format` | 10-30 seconds | 2 minutes |

**Network Issues**: Installation commands may encounter `ReadTimeoutError` due to PyPI connectivity. If this occurs:
1. Wait 1-2 minutes
2. Retry the exact same command
3. May require 3-5 retry attempts

### Task Delegation with Sub-Agents

Claude Code provides specialized sub-agents via the `task` tool. Use them proactively:

**When to use the `explore` agent:**
- Questions requiring codebase understanding or synthesis
- Multi-step searches requiring analysis
- When you want a summarized answer, not raw grep/glob results
- Examples: "How does authentication work?", "Where are API endpoints defined?"

**When to use the `task` agent:**
- Executing commands with verbose output (tests, builds, lints, dependency installs)
- Returns brief summary on success, full output on failure
- Keeps main context clean by minimizing successful output

**When to use direct tools (grep/glob):**
- Simple, targeted single searches where you know what to find
- Need results immediately in your context
- Looking for something specific, not discovering something unknown

**Parallel searches** - Call multiple grep/glob in ONE response:
```python
# Good: Parallel search calls
grep(pattern="function handleSubmit", glob="*.ts")
grep(pattern="interface FormData", glob="*.ts")
glob(pattern="**/*.tsx")
```

### Working Patterns

**Initial exploration:**
1. Use `explore` agent for codebase questions: "What does this module do?"
2. Use grep/glob for targeted searches: "Find all test files"
3. View key files identified: config files, main modules

**Making changes:**
1. Always validate first: run relevant fast tests to establish baseline
2. Make minimal, surgical changes
3. Test immediately after changes: `pytest tests/hyrax/test_<relevant>.py`
4. Format and lint: `ruff format . && ruff check --fix .`
5. Run full validation: `pytest -m "not slow"` (NEVER CANCEL, 10+ min timeout)
6. Run pre-commit: `pre-commit run --all-files` (NEVER CANCEL, 15+ min timeout)

**Common validation workflow:**
```bash
# Quick format/lint (30 seconds)
ruff format src/ tests/ && ruff check src/ tests/

# Fast tests (2-5 minutes, NEVER CANCEL)
pytest -m "not slow"

# Pre-commit (3-8 minutes, NEVER CANCEL)
pre-commit run --all-files
```

### Manual Validation After Changes

After making code changes, ALWAYS run these validation scenarios:

1. **CLI functionality**: `hyrax --help` and `hyrax --version` ensure CLI works
2. **Import test**: `python -c "import hyrax; h = hyrax.Hyrax(); print('Success')"`
3. **Configuration loading**: Verify config loads correctly
4. **Verb functionality**: Test relevant verbs like `hyrax train --help`

### Important Notes for Claude Code

**Batch editing**: Use the `edit` tool multiple times in a single response for:
- Renaming variables across multiple locations in the same file
- Editing non-overlapping blocks in the same or different files
- Applying the same pattern across multiple files

**Configuration system pitfalls**:
- Use `ConfigDict` instead of regular dict to catch missing defaults at runtime
- All config keys MUST have defaults in `hyrax_default_config.toml`
- Config is immutable after creation - no runtime mutations allowed

**Model interface requirements**:
- Models MUST implement: `forward()`, `train_step()`, `prepare_inputs()`
- Note: `to_tensor()` is deprecated, use `prepare_inputs()` instead
- Use `@hyrax_model` decorator for auto-registration

**Testing requirements**:
- Mark long-running tests with `@pytest.mark.slow`
- Fast tests (<5 min) run in pre-commit and CI
- Slow tests (>5 min) run separately
- Always run fast tests after changes: `pytest -m "not slow"`

**Pre-commit hooks include**:
- ruff linting and formatting
- pytest fast tests (not slow)
- sphinx documentation build
- jupyter notebook conversion
- Custom hook preventing note-to-self comments

## Key File Locations

Reference [HYRAX_GUIDE.md](./HYRAX_GUIDE.md#repository-structure) for full structure. Quick access:

```
src/hyrax/
  ├── hyrax.py                      # Main Hyrax class
  ├── config_utils.py               # ConfigManager, ConfigDict
  ├── plugin_utils.py               # get_or_load_class() for dynamic loading
  ├── train.py                      # Training orchestration
  ├── pytorch_ignite.py             # Dataset, model, dataloader setup
  ├── hyrax_default_config.toml     # Default configuration
  ├── models/model_registry.py      # Model registration, @hyrax_model
  ├── data_sets/data_set_registry.py # Dataset registration
  ├── verbs/verb_registry.py        # Verb registration, @hyrax_verb
  ├── config_schemas/               # Pydantic validation schemas
  └── vector_dbs/                   # ChromaDB, Qdrant implementations

src/hyrax_cli/main.py               # CLI entry point

tests/hyrax/
  ├── conftest.py                   # Shared test fixtures
  └── test_e2e.py                   # End-to-end integration tests

.github/
  ├── copilot-instructions.md       # GitHub Copilot instructions
  └── workflows/                    # CI/CD pipelines
```

## Common Pitfalls and Solutions

**Pitfall**: Forgetting to activate virtual environment
- **Solution**: Always check with `which python` or `pip list | grep hyrax`

**Pitfall**: Tests failing due to network issues during fixture download
- **Solution**: Tests use Pooch for reproducible downloads from Zenodo - retry if network fails

**Pitfall**: Pre-commit hooks not running
- **Solution**: Ensure `pre-commit install` was run after `pip install`

**Pitfall**: Config key not found errors
- **Solution**: Add missing key to `hyrax_default_config.toml` with sensible default

**Pitfall**: Model not registering
- **Solution**: Ensure `@hyrax_model("ModelName")` decorator is present and file is imported

**Pitfall**: Verb not appearing in CLI
- **Solution**: Ensure `@hyrax_verb("verb_name")` decorator is present and verb is imported

## Quick Command Reference

See [HYRAX_GUIDE.md](./HYRAX_GUIDE.md#essential-commands) for full command reference.

```bash
# Development setup
conda create -n hyrax python=3.10 && conda activate hyrax
cd hyrax && echo 'y' | bash .setup_dev.sh      # NEVER CANCEL: 20+ min timeout

# Quick validation (run after changes)
ruff format src/ tests/ && ruff check src/ tests/    # 30 seconds
pytest -m "not slow"                                  # NEVER CANCEL: 10+ min
pre-commit run --all-files                            # NEVER CANCEL: 15+ min

# Specific tests
pytest tests/hyrax/test_config_utils.py              # Single file
pytest tests/hyrax/test_infer.py::test_infer_basic  # Single test

# CLI verification
hyrax --help && hyrax --version                      # Verify CLI works
```

