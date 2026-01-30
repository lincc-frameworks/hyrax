---
name: Hyrax Configuration System
description: Essential guide for working with Hyrax's TOML configuration system and key invariants
version: 1.0.0
tags: configuration, config, toml, pydantic
---

# Hyrax Configuration System

This skill provides essential guidance on Hyrax's configuration system, focusing on key invariants and common patterns.

## When to Use

Use this skill when:
- Working with Hyrax configuration files
- Encountering "config key not found" errors
- Adding new configuration parameters
- Debugging configuration issues

## Configuration System Invariants

### Key Principles (MUST FOLLOW)

1. **All keys need defaults**: Every config key MUST have a default in `hyrax_default_config.toml`
2. **Config is immutable**: No runtime mutations allowed after creation
3. **Hierarchical merging**: User config merges with defaults
4. **Access via Hyrax object**: Get config from `hyrax.config`, not by creating ConfigDict directly

### Configuration Files

**Default config**: `src/hyrax/hyrax_default_config.toml`
```toml
# All possible keys with sensible defaults
[model]
name = "HyraxAutoencoderV2"
latent_dim = 128
learning_rate = 0.001

[data]
name = "DownloadedLSSTDataset"
batch_size = 32
num_workers = 4

[training]
epochs = 100
device = "cuda"
```

**User config**: `config.toml`
```toml
# Override only what you need
[model]
name = "MyCustomModel"
latent_dim = 256

[training]
epochs = 50
```

**Result**: User config merged with defaults automatically.

## Using Configuration

### Correct Pattern

```python
from hyrax import Hyrax

# Load config through Hyrax
hyrax = Hyrax(config_file="config.toml")
config = hyrax.config  # Immutable, merged config

# Read values
model_name = config["model"]["name"]
learning_rate = config["model"]["learning_rate"]

# Config is read-only - this is intentional
# config["model"]["name"] = "NewModel"  # This will fail!
```

### Why Immutable?

- Ensures reproducibility
- Config serves as experiment record
- Prevents accidental modifications
- Avoids hard-to-debug state changes

## Adding New Configuration Keys

### Step-by-Step Process

#### 1. Add to Default Config

File: `src/hyrax/hyrax_default_config.toml`

```toml
[model]
name = "HyraxAutoencoderV2"
latent_dim = 128
new_parameter = 10  # ← ADD THIS with sensible default
```

**CRITICAL**: EVERY key must have a default. Use `false` for optional features.

#### 2. Use in Code

```python
# Access through hyrax.config
new_value = config["model"]["new_parameter"]
```

## The `key = false` Convention

**Convention**: `key = false` in TOML means the key is optional and evaluates to `None` in Python.

```toml
[model]
optional_param = false  # Means "None" or "not set"

[data]
pretrained_weights = false  # Optional path
```

```python
# In Python
if config["model"]["optional_param"]:
    use_param(config["model"]["optional_param"])
# If false in TOML, this branch is skipped
```

## Common Configuration Errors

### Error 1: Config Key Not Found

**Error**: `ConfigKeyError: Key 'new_param' not found in config`

**Solution**: Add to `src/hyrax/hyrax_default_config.toml`:
```toml
[section]
new_param = default_value  # Or false if optional
```

### Error 2: Attempting Mutation

**Error**: `AttributeError: ConfigDict is immutable`

**Solution**: Config is intentionally read-only. This is by design for reproducibility.

## Best Practices

1. **Always provide defaults**: Every key must be in `hyrax_default_config.toml`
2. **Use `key = false` for optional features**: This clearly indicates optional parameters
3. **Access via Hyrax object**: Use `hyrax.config`, don't create ConfigDict directly
4. **Read-only access**: Never try to modify config after creation
5. **Document new keys**: Add descriptions to HYRAX_GUIDE.md

## Related Skills

- For using config in components, see: **Adding Hyrax Components**
- For overall workflow, see: **Hyrax Development Workflow**
- For testing config, see: **Hyrax Testing Strategy**

## References

- Configuration guide: [HYRAX_GUIDE.md](../../HYRAX_GUIDE.md#configuration-system)
- Default config: `src/hyrax/hyrax_default_config.toml`
- ConfigDict implementation: `src/hyrax/config_utils.py`
