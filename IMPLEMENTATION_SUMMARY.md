# Data Location Path Resolution Implementation

## Summary

This implementation adds automatic path resolution for `data_location` values in the `DataProvider` class.

## Changes Made

### 1. Modified `src/hyrax/data_sets/data_provider.py`

- Added `from pathlib import Path` import
- Added path resolution logic in the `prepare_datasets()` method
- Resolution happens right after reading `data_location` from the config
- Uses `Path(data_location).expanduser().resolve()` to:
  - Expand `~` to home directory
  - Resolve relative paths to absolute paths
  - Resolve symlinks
- Writes resolved paths back to:
  - The local `dataset_definition` dict
  - The main `config["data_request"]` dict

### 2. Created `tests/hyrax/test_data_location_path_resolution.py`

New comprehensive test suite with 6 tests:
- `test_relative_path_resolution()`: Verifies relative paths are converted to absolute
- `test_absolute_path_unchanged()`: Ensures absolute paths remain unchanged
- `test_none_data_location()`: Handles None/missing data_location gracefully
- `test_tilde_path_expansion()`: Verifies `~` is expanded to home directory
- `test_multiple_datasets_path_resolution()`: Tests multiple datasets are handled independently
- `test_persisted_config_has_resolved_paths()`: Confirms persisted config contains resolved paths

### 3. Updated `tests/hyrax/test_data_provider.py`

Modified `test_primary_or_first_dataset()` to expect absolute paths instead of relative paths.

## Code Example

### Before
```toml
[data_request.train.data]
dataset_class = "DatasetClassName"
data_location = "./data"
```

### After (in runtime_config.toml)
```toml
[data_request.train.data]
dataset_class = "DatasetClassName"
data_location = "/home/runner/work/hyrax/hyrax/data"
```

## Testing

All tests pass:
- 6 new tests in `test_data_location_path_resolution.py` ✓
- 26 existing tests in `test_data_provider.py` ✓
- Code formatting and linting with ruff ✓

## Design Decisions

1. **Localized to data_provider module**: All changes are within `data_provider.py` - no changes needed in `verbs/` directory
2. **Minimal changes**: Only 11 lines of code added
3. **No required calls from verbs**: Resolution happens automatically during `DataProvider.__init__()`
4. **Preserves None values**: Datasets that don't require `data_location` continue to work
5. **Consistent with ConfigManager**: Uses same pattern as `ConfigManager._resolve_config_paths()`
