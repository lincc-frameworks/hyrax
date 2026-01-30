---
name: Hyrax Testing Strategy
description: Comprehensive guide for testing Hyrax code with pytest, including slow/fast markers, parallel execution, and fixtures
version: 1.0.0
tags: testing, pytest, validation
---

# Hyrax Testing Strategy

This skill provides detailed guidance on Hyrax's testing strategy, including test markers, execution modes, and common patterns.

## When to Use

Use this skill when:
- Running tests on Hyrax code changes
- Deciding which tests to run (fast vs slow)
- Debugging test failures
- Adding new tests to the codebase
- Understanding fixture usage and test patterns

## Test Organization

### Fast Tests vs Slow Tests

Hyrax uses pytest markers to categorize tests by execution time:

**Fast Tests** (< 5 minutes total):
- Marked WITHOUT `@pytest.mark.slow`
- Run in pre-commit hooks and CI
- Should complete in 2-5 minutes total
- Use mocks/fixtures to avoid heavy I/O

**Slow Tests** (> 5 minutes individual):
- Marked WITH `@pytest.mark.slow`
- Include end-to-end integration tests
- Run separately from fast tests
- May involve network I/O, large datasets

### Test Execution Commands

```bash
# Fast tests only (DEFAULT - use after code changes)
pytest -m "not slow"
# Duration: 2-5 minutes, NEVER CANCEL (wait 10+ minutes)

# Fast tests with parallel execution
pytest -n auto -m "not slow"
# Uses all CPU cores, faster completion

# Fast tests with coverage
pytest -n auto --cov=./src --cov-report=html -m "not slow"
# Generates HTML coverage report in htmlcov/

# Slow tests only
pytest -m slow
# Duration: 10-20 minutes, NEVER CANCEL (wait 30+ minutes)

# All tests (fast + slow)
pytest
# Duration: 15-25 minutes, NEVER CANCEL (wait 45+ minutes)

# Single test file
pytest tests/hyrax/test_config_utils.py

# Single test function
pytest tests/hyrax/test_infer.py::test_infer_basic

# Verbose output for debugging
pytest -v -s tests/hyrax/test_specific.py
```

## Marking New Tests

### When to Mark as Slow

Mark a test with `@pytest.mark.slow` if:
- Individual test takes > 5 minutes
- Test downloads large datasets
- Test performs heavy computation
- Test runs full end-to-end workflows

### How to Mark Tests

```python
import pytest

# Fast test (no marker needed)
def test_config_loading():
    """Test configuration loading - fast."""
    config = ConfigDict()
    assert config is not None

# Slow test (add marker)
@pytest.mark.slow
def test_full_training_pipeline():
    """Test complete training workflow - slow."""
    # This test trains a model end-to-end
    result = run_full_training()
    assert result.success

# Multiple markers
@pytest.mark.slow
@pytest.mark.integration
def test_e2e_workflow():
    """End-to-end integration test."""
    pass
```

## Test Fixtures

### Common Fixtures

Fixtures are defined in `tests/hyrax/conftest.py`:

```python
# Example fixture usage
def test_with_sample_data(sample_dataset):
    """Test using sample data fixture."""
    assert len(sample_dataset) > 0

def test_with_config(default_config):
    """Test using default config fixture."""
    assert default_config["model"]["name"] is not None
```

### Data Download with Pooch

Tests use Pooch for reproducible downloads from Zenodo:
- Downloads are cached locally
- Network failures should trigger retry, not test failure
- DOIs provide version-locked data access

## Network Retry Strategy

### Handling Network Failures

**PyPI Installation**:
```bash
# If pip install fails with ReadTimeoutError:
# 1. Wait 1-2 minutes
# 2. Retry exact same command
# 3. May need 3-5 attempts
pip install -e .'[dev]'
```

**Test Data Downloads**:
- Tests using Pooch may fail on poor network
- Retry test execution if network-related failure
- Local cache prevents re-downloading on retry

## Parallel Testing

### Using pytest-xdist

```bash
# Automatic CPU detection
pytest -n auto -m "not slow"

# Explicit worker count
pytest -n 4 -m "not slow"

# With verbose output
pytest -n auto -v -m "not slow"
```

**Benefits**:
- Faster test execution
- Better CPU utilization
- Scales with available cores

**Limitations**:
- May hide race conditions
- Output interleaving in verbose mode
- Some fixtures may not be thread-safe

## Test Execution Timeouts

**CRITICAL: Never cancel tests prematurely**

| Test Command | Expected Duration | Minimum Timeout |
|-------------|-------------------|----------------|
| `pytest -m "not slow"` | 2-5 min | 10+ minutes |
| `pytest -m slow` | 10-20 min | 30+ minutes |
| `pytest` (all) | 15-25 min | 45+ minutes |
| Single test file | 10 sec - 2 min | 5 minutes |

## Debugging Test Failures

### Common Failure Patterns

**Config Key Not Found**:
```python
# Problem: Missing config key
# Solution: Add to src/hyrax/hyrax_default_config.toml
```

**Import Errors**:
```bash
# Verify package installed in development mode
pip list | grep hyrax
# Should show: hyrax <version> /path/to/src
```

**Network Timeouts**:
```bash
# Retry test if Pooch download fails
pytest tests/hyrax/test_that_failed.py
```

**Fixture Not Found**:
```python
# Check tests/hyrax/conftest.py for available fixtures
# Import fixture if defined in different conftest.py
```

### Debugging Commands

```bash
# Run with full output
pytest -v -s tests/hyrax/test_file.py

# Run with pdb on failure
pytest --pdb tests/hyrax/test_file.py

# Show local variables on failure
pytest -l tests/hyrax/test_file.py

# Stop on first failure
pytest -x tests/hyrax/test_file.py
```

## Pre-commit Testing

Pre-commit hooks run fast tests automatically:
```bash
# Manual pre-commit execution
pre-commit run --all-files
# Duration: 3-8 minutes, NEVER CANCEL (wait 15+ minutes)

# Only runs pytest fast tests (not slow)
# If fast tests fail, commit is blocked
```

## Writing New Tests

### Test Structure

```python
import pytest
from hyrax import Hyrax

class TestMyFeature:
    """Test suite for my feature."""
    
    def test_basic_functionality(self):
        """Test basic feature behavior."""
        result = my_feature()
        assert result == expected
    
    @pytest.mark.slow
    def test_full_workflow(self):
        """Test complete workflow - slow."""
        # End-to-end test
        pass
    
    def test_error_handling(self):
        """Test error cases."""
        with pytest.raises(ValueError):
            my_feature(invalid_input)
```

### Best Practices

1. **Keep fast tests fast**: Mock heavy operations
2. **Use descriptive names**: Test name explains what's tested
3. **One assertion per test**: Focus on single behavior
4. **Use fixtures**: Avoid code duplication
5. **Test edge cases**: Don't just test happy path
6. **Mark slow tests**: Use `@pytest.mark.slow` appropriately

## Integration with CI/CD

### CI Workflow

1. **Pre-commit CI**: Runs fast tests on every push
2. **Testing and Coverage**: Full test suite with coverage reports
3. **Smoke Tests**: Daily scheduled tests
4. **Performance**: ASV benchmarks track performance

See `.github/workflows/testing-and-coverage.yml` for details.

## Related Skills

- For overall workflow, see: **Hyrax Development Workflow**
- For adding tests, see: **Adding Hyrax Components**
- For config in tests, see: **Hyrax Configuration System**

## References

- Full testing guide: [HYRAX_GUIDE.md](../../HYRAX_GUIDE.md#testing)
- Test fixtures: `tests/hyrax/conftest.py`
- CI workflows: `.github/workflows/`
