---
name: Hyrax Configuration System
description: Comprehensive guide for working with Hyrax's TOML configuration system, ConfigDict, and Pydantic validation
version: 1.0.0
tags: configuration, config, toml, pydantic
---

# Hyrax Configuration System

This skill provides detailed guidance on Hyrax's configuration system, including ConfigDict usage, default requirements, immutability rules, and common pitfalls.

## When to Use

Use this skill when:
- Working with Hyrax configuration files
- Encountering "config key not found" errors
- Adding new configuration parameters
- Understanding ConfigDict vs regular dict
- Debugging configuration validation issues
- Understanding the `key = false` convention

## Configuration System Overview

### Key Principles

1. **All keys need defaults**: Every config key MUST have a default in `hyrax_default_config.toml`
2. **Config is immutable**: No runtime mutations allowed after creation
3. **Use ConfigDict, not dict**: ConfigDict catches missing defaults at runtime
4. **Pydantic validation**: Schemas validate structure and types
5. **Hierarchical merging**: User config merges with defaults

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

**Result**: User config merged with defaults
```toml
[model]
name = "MyCustomModel"        # User override
latent_dim = 256              # User override
learning_rate = 0.001         # Default

[data]
name = "DownloadedLSSTDataset"  # Default
batch_size = 32               # Default
num_workers = 4               # Default

[training]
epochs = 50                   # User override
device = "cuda"               # Default
```

## ConfigDict vs Dict

### Why ConfigDict?

**Regular dict** (DON'T USE):
```python
config = {"model": {"name": "MyModel"}}

# Silent failure - returns None
value = config.get("nonexistent_key", None)

# KeyError only if you access directly
value = config["nonexistent_key"]  # KeyError
```

**ConfigDict** (USE THIS):
```python
from hyrax.config_utils import ConfigDict

config = ConfigDict({"model": {"name": "MyModel"}})

# Fails immediately with helpful error
value = config["nonexistent_key"]  # Raises ConfigKeyError

# Also fails for nested keys
value = config["model"]["nonexistent_param"]  # Raises ConfigKeyError
```

### Creating ConfigDict

```python
from hyrax.config_utils import ConfigDict, ConfigManager

# Method 1: From file
config = ConfigManager.load_config("config.toml")
# Returns ConfigDict with defaults merged

# Method 2: From dict
config_dict = {"model": {"name": "MyModel"}}
config = ConfigDict(config_dict)

# Method 3: From Hyrax
from hyrax import Hyrax
hyrax = Hyrax(config_file="config.toml")
config = hyrax.config  # ConfigDict
```

## Configuration Immutability

### Why Immutable?

- Ensures reproducibility
- Prevents accidental modifications
- Config serves as experiment record
- Avoids hard-to-debug state changes

### Correct Usage

```python
# ✅ CORRECT: Read-only access
model_name = config["model"]["name"]
learning_rate = config["training"]["learning_rate"]

# ✅ CORRECT: Create new config for modifications
new_config_dict = config.to_dict()
new_config_dict["model"]["name"] = "NewModel"
new_config = ConfigDict(new_config_dict)

# ✅ CORRECT: Modify before creating ConfigDict
config_dict = {"model": {"name": "Model1"}}
config_dict["model"]["name"] = "Model2"
config = ConfigDict(config_dict)  # Now immutable
```

### Incorrect Usage

```python
# ❌ WRONG: Attempting to modify after creation
config["model"]["name"] = "NewModel"  # Error!

# ❌ WRONG: Trying to add new keys
config["new_section"] = {}  # Error!

# ❌ WRONG: Modifying nested dicts
config["model"]["new_param"] = 100  # Error!
```

## Adding New Configuration Keys

### Step-by-Step Process

#### 1. Add to Default Config

File: `src/hyrax/hyrax_default_config.toml`

```toml
# Add new key with sensible default
[model]
name = "HyraxAutoencoderV2"
latent_dim = 128
new_parameter = 10  # ← ADD THIS
```

**CRITICAL**: EVERY key must have a default, even if it's `false` (see next section).

#### 2. Update Pydantic Schema (Optional but Recommended)

File: `src/hyrax/config_schemas/model_config.py`

```python
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Pydantic schema for model configuration."""
    name: str = Field(description="Model name")
    latent_dim: int = Field(gt=0, description="Latent dimension")
    new_parameter: int = Field(default=10, description="New parameter")  # ← ADD THIS
```

#### 3. Use in Code

```python
def my_function(config):
    """Use new config parameter."""
    new_value = config["model"]["new_parameter"]
    # ... use new_value
```

#### 4. Document in HYRAX_GUIDE.md

Add to configuration reference:
```markdown
### Model Configuration

- `new_parameter` (int, default: 10): Description of what this parameter does
```

## The `key = false` Convention

### What Does `key = false` Mean?

**Convention**: `key = false` in TOML means the key is optional and defaults to `None` in Python.

**In TOML**:
```toml
[model]
optional_param = false  # Means "None" or "not set"
```

**In Python**:
```python
# When key = false in TOML
config["model"]["optional_param"]  # Returns None

# Check if set
if config["model"]["optional_param"]:
    use_param(config["model"]["optional_param"])
else:
    # Use default behavior
    pass
```

### When to Use `key = false`

```toml
# Use for optional features
[model]
use_batch_norm = false  # Optional: batch normalization
dropout_rate = false    # Optional: dropout

# Use for optional paths
[data]
pretrained_weights = false  # Optional: path to weights
custom_transform = false     # Optional: custom transform

# Use for optional integrations
[logging]
mlflow_uri = false       # Optional: MLflow tracking
wandb_project = false    # Optional: Weights & Biases
```

### User Overrides

User can enable by setting to actual value:
```toml
# User config.toml
[model]
use_batch_norm = true
dropout_rate = 0.5

[data]
pretrained_weights = "/path/to/weights.pt"
```

## Configuration Hierarchy

### Merge Order

1. **Built-in defaults**: `hyrax_default_config.toml`
2. **Component defaults**: `<component>_default_config.toml`
3. **External plugin defaults**: `<package>/default_config.toml`
4. **User config**: `config.toml`

Later configs override earlier ones.

### Example Merge

**hyrax_default_config.toml**:
```toml
[model]
name = "HyraxAutoencoderV2"
latent_dim = 128
learning_rate = 0.001
```

**my_model_default_config.toml**:
```toml
[model]
latent_dim = 256  # Component default overrides
custom_param = 42
```

**config.toml** (user):
```toml
[model]
name = "MyModel"  # User overrides
learning_rate = 0.0001
```

**Final merged config**:
```toml
[model]
name = "MyModel"          # From user
latent_dim = 256          # From component
learning_rate = 0.0001    # From user
custom_param = 42         # From component
```

## Pydantic Validation

### Why Pydantic?

- Type checking at runtime
- Helpful error messages
- Schema documentation
- Validation rules (gt, lt, regex, etc.)

### Validation Schemas

Location: `src/hyrax/config_schemas/`

```python
from pydantic import BaseModel, Field, validator

class ModelConfig(BaseModel):
    """Model configuration schema."""
    
    name: str = Field(description="Model name")
    latent_dim: int = Field(gt=0, description="Latent dimension must be positive")
    learning_rate: float = Field(gt=0, lt=1, description="Learning rate in (0, 1)")
    
    @validator("name")
    def name_not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Model name cannot be empty")
        return v
```

### Using Validation

```python
from hyrax.config_schemas import ModelConfig

# Validate config section
model_config = ModelConfig(**config["model"])

# Pydantic raises ValidationError if invalid
# Example errors:
# - "latent_dim must be greater than 0"
# - "learning_rate must be greater than 0 and less than 1"
# - "name cannot be empty"
```

## Common Configuration Errors

### Error 1: Config Key Not Found

**Error**:
```
ConfigKeyError: Key 'new_param' not found in config
```

**Cause**: Key not in `hyrax_default_config.toml`

**Solution**:
1. Add to `src/hyrax/hyrax_default_config.toml`:
   ```toml
   [section]
   new_param = default_value
   ```
2. If optional, use `new_param = false`

### Error 2: Type Mismatch

**Error**:
```
ValidationError: latent_dim: value is not a valid integer
```

**Cause**: Config value has wrong type

**Solution**:
Fix in config file:
```toml
# ❌ WRONG
[model]
latent_dim = "128"  # String

# ✅ CORRECT
[model]
latent_dim = 128    # Integer
```

### Error 3: Missing Section

**Error**:
```
ConfigKeyError: Key 'model' not found in config
```

**Cause**: Entire config section missing

**Solution**:
Add section to `hyrax_default_config.toml`:
```toml
[model]
name = "HyraxAutoencoderV2"
latent_dim = 128
```

### Error 4: Attempting Mutation

**Error**:
```
AttributeError: ConfigDict is immutable
```

**Cause**: Trying to modify ConfigDict after creation

**Solution**:
Create new config instead:
```python
# Convert to dict, modify, create new ConfigDict
new_dict = config.to_dict()
new_dict["model"]["name"] = "NewModel"
new_config = ConfigDict(new_dict)
```

## Configuration Best Practices

### 1. Always Use ConfigDict

```python
# ✅ CORRECT
from hyrax.config_utils import ConfigDict
config = ConfigDict(config_dict)

# ❌ WRONG
config = config_dict  # Just a dict
```

### 2. Provide Sensible Defaults

```toml
# ✅ CORRECT: Reasonable defaults that work out of box
[training]
batch_size = 32
learning_rate = 0.001
epochs = 100

# ❌ WRONG: Defaults that require user to change
[training]
batch_size = 1000000  # Unrealistic
learning_rate = 1.0   # Too high
epochs = 0            # Won't train
```

### 3. Use `key = false` for Optional Features

```toml
# ✅ CORRECT: Optional features default to false
[logging]
mlflow_uri = false
wandb_project = false

# ❌ WRONG: Optional features with dummy values
[logging]
mlflow_uri = ""           # Empty string is confusing
wandb_project = "none"    # "none" might be actual project name
```

### 4. Validate Early

```python
# ✅ CORRECT: Validate config on load
config = ConfigManager.load_config("config.toml")
# Validation happens here - fail fast

# ❌ WRONG: Validate late in processing
# ... hours of processing ...
# Finally try to use config key - error!
```

### 5. Document Configuration

```python
class MyModel:
    """My model.
    
    Configuration:
        model.name: Model identifier
        model.latent_dim: Latent space dimensionality (must be > 0)
        model.learning_rate: Learning rate for optimizer (0 < lr < 1)
    """
```

## Working with External Plugins

### Plugin Config Loading

```toml
# User config.toml
[model]
name = "external_package.CustomModel"  # External model
```

Hyrax will:
1. Import `external_package.CustomModel`
2. Look for `external_package/default_config.toml`
3. Merge with hyrax defaults and user config

### Plugin Default Config

Location: `external_package/default_config.toml`

```toml
[model]
latent_dim = 512
custom_param = "value"

[external_package_settings]
plugin_option = true
```

## Debugging Configuration

### Print Current Config

```python
from hyrax import Hyrax

hyrax = Hyrax(config_file="config.toml")

# Print entire config
print(hyrax.config.to_dict())

# Print specific section
print(hyrax.config["model"])

# Pretty print
import json
print(json.dumps(hyrax.config.to_dict(), indent=2))
```

### Check Default Values

```bash
# View default config
cat src/hyrax/hyrax_default_config.toml

# Search for specific key
grep -n "key_name" src/hyrax/hyrax_default_config.toml
```

### Validate Config File

```python
# Load and validate
from hyrax.config_utils import ConfigManager

try:
    config = ConfigManager.load_config("config.toml")
    print("Config valid!")
except Exception as e:
    print(f"Config error: {e}")
```

## Related Skills

- For using config in components, see: **Adding Hyrax Components**
- For overall workflow, see: **Hyrax Development Workflow**
- For testing config, see: **Hyrax Testing Strategy**

## References

- Configuration guide: [HYRAX_GUIDE.md](../../HYRAX_GUIDE.md#configuration-system)
- Default config: `src/hyrax/hyrax_default_config.toml`
- ConfigDict implementation: `src/hyrax/config_utils.py`
- Pydantic schemas: `src/hyrax/config_schemas/`
