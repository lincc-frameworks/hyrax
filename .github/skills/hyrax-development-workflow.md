---
name: Hyrax Development Workflow
description: Guide through the complete development workflow for Hyrax: setup → code → test → commit
version: 1.0.0
tags: development, workflow, setup, testing
---

# Hyrax Development Workflow

This skill provides structured guidance for the Hyrax development workflow, including setup, validation, and commit procedures.

## When to Use

Use this skill when:
- Setting up a new development environment for Hyrax
- Validating code changes before committing
- Running pre-commit hooks and ensuring code quality
- Unsure about which commands to run or their expected duration

## Workflow Steps

### 1. Environment Setup

```bash
# Create and activate virtual environment
conda create -n hyrax python=3.10 && conda activate hyrax

# Clone and navigate to repository
git clone https://github.com/lincc-frameworks/hyrax.git && cd hyrax

# Automated setup (RECOMMENDED)
echo 'y' | bash .setup_dev.sh
# Duration: 5-15 minutes, NEVER CANCEL (wait at least 20 minutes)
# Installs with pip install -e .'[dev]' and sets up pre-commit hooks
# Prompts for system install if no venv - respond 'y'

# Alternative: Manual installation
pip install -e .'[dev]' && pre-commit install
# Duration: 5-15 minutes, NEVER CANCEL (wait at least 20 minutes)
```

**Network Issues**: Installation commands may encounter `ReadTimeoutError` from PyPI:
1. Wait 1-2 minutes
2. Retry the exact same command
3. May require 3-5 retry attempts to succeed

### 2. Making Code Changes

1. **Always validate first**: Run relevant fast tests to establish baseline
2. **Make minimal, surgical changes**: Change only what's necessary
3. **Test immediately**: Run targeted tests after each change

### 3. Validation Workflow (CRITICAL)

**ALWAYS run these steps after making code changes:**

```bash
# Step 1: Format and lint (30 seconds)
ruff format . && ruff check --fix .

# Step 2: Fast tests (2-5 minutes, NEVER CANCEL - wait at least 10 minutes)
pytest -m "not slow"

# Step 3: Pre-commit hooks (3-8 minutes, NEVER CANCEL - wait at least 15 minutes)
pre-commit run --all-files
```

**Manual validation scenarios** (when automated tests aren't sufficient):
```bash
# Verify CLI functionality
hyrax --help && hyrax --version

# Test import and basic construction
python -c "import hyrax; h = hyrax.Hyrax(); print('Success')"

# Test relevant verbs
hyrax <verb> --help
```

### 4. Command Timeout Reference

**CRITICAL: Never cancel these commands prematurely**

| Command | Typical Duration | Minimum Timeout |
|---------|------------------|----------------|
| `bash .setup_dev.sh` | 5-15 min | 20+ minutes |
| `pip install -e .'[dev]'` | 5-15 min | 20+ minutes |
| `pytest -m "not slow"` | 2-5 min | 10+ minutes |
| `pytest` (all tests) | 15-25 min | 45+ minutes |
| `pytest -m slow` | 10-20 min | 30+ minutes |
| `pre-commit run --all-files` | 3-8 min | 15+ minutes |
| `sphinx-build` (docs) | 2-4 min | 10+ minutes |
| `ruff format/check` | 10-30 sec | 2 minutes |

### 5. Committing Changes

- Use `report_progress` tool to commit and push changes
- Do NOT use `git commit` or `git push` directly
- Review files committed by `report_progress`
- Use `.gitignore` to exclude build artifacts, temp files, dependencies

### 6. Pre-commit Hooks

The pre-commit hooks automatically run:
- ruff linting and formatting
- pytest fast tests (not slow)
- sphinx documentation build
- jupyter notebook conversion
- Custom hook: prevents note-to-self comments

## Common Issues

### Setup Issues
- **ReadTimeoutError during pip install**: Retry 3-5 times with 1-2 min waits
- **CLI not found after install**: Check with `pip list | grep hyrax`
- **Pre-commit not running**: Ensure `pre-commit install` was executed
- **Import errors**: Verify `pip install -e .'[dev]'` completed successfully

### Validation Issues
- **Tests canceled prematurely**: Commands need full timeout duration
- **Network failures in tests**: Tests use Pooch for downloads - retry if network fails
- **Pre-commit failures**: Address each failure individually, don't skip

## Best Practices

1. **Always use virtual environment**: Avoid system Python conflicts
2. **Run validation early and often**: Catch issues before they compound
3. **Never cancel long-running commands**: They need time to complete
4. **Test incrementally**: Run targeted tests after each change
5. **Format before testing**: Fix style issues first with ruff

## Related Skills

- For testing details, see: **Hyrax Testing Strategy**
- For adding components, see: **Adding Hyrax Components**
- For config changes, see: **Hyrax Configuration System**

## References

- Full details: [HYRAX_GUIDE.md](../../HYRAX_GUIDE.md)
- Claude-specific: [CLAUDE.md](../../CLAUDE.md)
- Copilot-specific: [.github/copilot-instructions.md](../copilot-instructions.md)
