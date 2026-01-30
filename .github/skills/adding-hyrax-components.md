---
name: Adding Hyrax Components
description: Step-by-step guide for adding new models, datasets, and verbs to Hyrax with proper registration
version: 1.0.0
tags: development, models, datasets, verbs, plugins
---

# Adding Hyrax Components

This skill provides structured guidance for adding new components to Hyrax, including models, datasets, and verbs with proper registration and decorator usage.

## When to Use

Use this skill when:
- Adding a new machine learning model to Hyrax
- Creating a new dataset class for data access
- Implementing a new CLI verb/command
- Troubleshooting component registration issues
- Understanding decorator requirements

## Adding a New Model

### Model Requirements

All Hyrax models MUST implement:
1. `forward()` - Forward pass through model
2. `train_step()` - Single training step logic
3. `prepare_inputs()` - Data preparation (replaces deprecated `to_tensor()`)

### Step-by-Step Process

#### 1. Create Model File

Location: `src/hyrax/models/my_model.py`

```python
import torch
import torch.nn as nn
from hyrax.models.model_registry import hyrax_model

@hyrax_model("MyModel")
class MyModel(nn.Module):
    """My custom model for astronomy tasks.
    
    This model implements <describe architecture>.
    """
    
    def __init__(self, config):
        """Initialize model.
        
        Args:
            config: Hyrax config dictionary
        """
        super().__init__()
        self.config = config
        
        # Build model layers
        self.encoder = nn.Sequential(
            nn.Linear(config["model"]["input_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, config["model"]["latent_dim"])
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.encoder(x)
    
    def train_step(self, batch, optimizer, device):
        """Single training step.
        
        Args:
            batch: Dictionary with batch data
            optimizer: PyTorch optimizer
            device: torch device
            
        Returns:
            Dictionary with 'loss' and optional metrics
        """
        # Prepare inputs
        inputs = self.prepare_inputs(batch, device)
        
        # Forward pass
        outputs = self(inputs)
        
        # Compute loss
        loss = compute_loss(outputs, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item()}
    
    def prepare_inputs(self, batch, device):
        """Prepare batch data for model input.
        
        Args:
            batch: Dictionary with batch data
            device: torch device
            
        Returns:
            Prepared input tensor
        """
        # Extract and prepare inputs
        images = batch["image"]
        return images.to(device)
```

#### 2. Register Model (Automatic)

The `@hyrax_model("MyModel")` decorator handles:
- Automatic registration in model registry
- Shape inference for model outputs
- CLI integration

#### 3. Add Default Config

Location: `src/hyrax/models/my_model_default_config.toml`

```toml
[model]
name = "MyModel"
input_dim = 1024
latent_dim = 128
learning_rate = 0.001
```

#### 4. Import in `__init__.py`

Location: `src/hyrax/models/__init__.py`

```python
from .my_model import MyModel

__all__ = [..., "MyModel"]
```

#### 5. Test the Model

```python
# tests/hyrax/models/test_my_model.py
import pytest
from hyrax import Hyrax
from hyrax.models import MyModel

def test_my_model_creation():
    """Test model can be created."""
    config = {"model": {"name": "MyModel", "input_dim": 100, "latent_dim": 32}}
    model = MyModel(config)
    assert model is not None

def test_my_model_forward():
    """Test forward pass."""
    config = {"model": {"name": "MyModel", "input_dim": 100, "latent_dim": 32}}
    model = MyModel(config)
    x = torch.randn(10, 100)
    output = model(x)
    assert output.shape == (10, 32)
```

#### 6. Use via CLI

```bash
# Create config.toml with:
# [model]
# name = "MyModel"
# input_dim = 1024
# latent_dim = 128

hyrax train -c config.toml
```

## Adding a New Dataset

### Dataset Requirements

All Hyrax datasets MUST:
1. Subclass `HyraxDataset` or `HyraxImageDataset`
2. Set `_name` class attribute (triggers auto-registration)
3. Implement: `__len__()`, `__getitem__()`, metadata interface

### Step-by-Step Process

#### 1. Create Dataset File

Location: `src/hyrax/data_sets/my_dataset.py`

```python
import torch
from pathlib import Path
from hyrax.data_sets.hyrax_image_dataset import HyraxImageDataset

class MyDataset(HyraxImageDataset):
    """Custom dataset for my astronomy data.
    
    This dataset loads <describe data source>.
    """
    
    # Auto-registration via _name attribute
    _name = "MyDataset"
    
    def __init__(self, config, split="train"):
        """Initialize dataset.
        
        Args:
            config: Hyrax config dictionary
            split: Data split (train/val/test)
        """
        super().__init__(config, split)
        
        # Load data file list
        self.data_dir = Path(config["data"]["root_dir"])
        self.file_list = self._load_file_list(split)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Get single data item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with data item
        """
        # Load image
        image_path = self.file_list[idx]
        image = self._load_image(image_path)
        
        # Apply transforms (handled by HyraxImageDataset)
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "label": self._get_label(idx),
            "metadata": self._get_metadata(idx)
        }
    
    def _load_file_list(self, split):
        """Load list of files for split."""
        split_file = self.data_dir / f"{split}.txt"
        with open(split_file) as f:
            return [line.strip() for line in f]
    
    def _load_image(self, path):
        """Load single image."""
        # Implementation depends on image format
        pass
    
    def _get_label(self, idx):
        """Get label for item."""
        # Return label if supervised, None if unsupervised
        pass
    
    def _get_metadata(self, idx):
        """Get metadata for item."""
        return {"index": idx, "path": str(self.file_list[idx])}
```

#### 2. Registration (Automatic)

Setting `_name = "MyDataset"` triggers automatic registration via `__init_subclass__`.

#### 3. Add Default Config

Location: `src/hyrax/data_sets/my_dataset_default_config.toml`

```toml
[data]
name = "MyDataset"
root_dir = "/path/to/data"
batch_size = 32
num_workers = 4
```

#### 4. Import in `__init__.py`

Location: `src/hyrax/data_sets/__init__.py`

```python
from .my_dataset import MyDataset

__all__ = [..., "MyDataset"]
```

#### 5. Test the Dataset

```python
# tests/hyrax/data_sets/test_my_dataset.py
import pytest
from hyrax.data_sets import MyDataset

def test_my_dataset_creation(tmp_path):
    """Test dataset can be created."""
    config = {"data": {"name": "MyDataset", "root_dir": str(tmp_path)}}
    dataset = MyDataset(config)
    assert dataset is not None

def test_my_dataset_length(tmp_path):
    """Test dataset length."""
    # Setup test data
    dataset = MyDataset(config)
    assert len(dataset) > 0

def test_my_dataset_getitem(tmp_path):
    """Test getting data item."""
    dataset = MyDataset(config)
    item = dataset[0]
    assert "image" in item
    assert "label" in item
```

## Adding a New Verb

### Verb Requirements

All Hyrax verbs MUST:
1. Use `@hyrax_verb("verb_name")` decorator
2. Implement `run()` and `run_cli()` methods
3. Implement `setup_parser(parser)` for CLI args
4. Set `add_parser_kwargs` for help text

### Step-by-Step Process

#### 1. Create Verb File

Location: `src/hyrax/verbs/my_verb.py`

```python
import argparse
from pathlib import Path
from hyrax.verbs.verb_registry import hyrax_verb
from hyrax import Hyrax

@hyrax_verb("myverb")
class MyVerb:
    """Custom verb for doing something useful.
    
    This verb performs <describe functionality>.
    """
    
    # Help text for CLI
    add_parser_kwargs = {
        "help": "Perform my custom operation",
        "description": "Detailed description of what this verb does."
    }
    
    @staticmethod
    def setup_parser(parser):
        """Setup argument parser for CLI.
        
        Args:
            parser: argparse.ArgumentParser
        """
        parser.add_argument(
            "-c", "--config",
            type=str,
            required=True,
            help="Path to configuration file"
        )
        parser.add_argument(
            "-o", "--output",
            type=str,
            help="Output directory"
        )
        parser.add_argument(
            "--option",
            action="store_true",
            help="Enable optional behavior"
        )
    
    @staticmethod
    def run(config, output_dir=None, option=False):
        """Execute verb logic.
        
        Args:
            config: Hyrax config dictionary
            output_dir: Optional output directory
            option: Optional flag
            
        Returns:
            Results dictionary
        """
        # Initialize Hyrax
        hyrax = Hyrax(config)
        
        # Perform verb logic
        results = perform_operation(hyrax, output_dir, option)
        
        # Save results
        if output_dir:
            save_results(results, output_dir)
        
        return results
    
    @staticmethod
    def run_cli(args):
        """Execute verb from CLI.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        # Load config
        config = load_config(args.config)
        
        # Run verb
        results = MyVerb.run(
            config,
            output_dir=args.output,
            option=args.option
        )
        
        # Print summary
        print(f"Operation complete: {results['summary']}")
        
        return 0
```

#### 2. Registration (Automatic)

The `@hyrax_verb("myverb")` decorator handles automatic CLI registration.

#### 3. Import in `__init__.py`

Location: `src/hyrax/verbs/__init__.py`

```python
from .my_verb import MyVerb

__all__ = [..., "MyVerb"]
```

#### 4. Test the Verb

```python
# tests/hyrax/verbs/test_my_verb.py
import pytest
from hyrax.verbs import MyVerb

def test_my_verb_run(default_config):
    """Test verb execution."""
    results = MyVerb.run(default_config)
    assert results is not None
    assert "summary" in results

def test_my_verb_cli(tmp_path, default_config):
    """Test CLI execution."""
    # Save config
    config_path = tmp_path / "config.toml"
    save_config(default_config, config_path)
    
    # Mock CLI args
    args = argparse.Namespace(
        config=str(config_path),
        output=str(tmp_path),
        option=True
    )
    
    # Run CLI
    exit_code = MyVerb.run_cli(args)
    assert exit_code == 0
```

#### 5. Use via CLI

```bash
# View help
hyrax myverb --help

# Execute verb
hyrax myverb -c config.toml -o output/ --option
```

## Common Registration Issues

### Model Not Registering

**Problem**: Model not available in CLI

**Solutions**:
1. Verify `@hyrax_model("ModelName")` decorator present
2. Check model imported in `src/hyrax/models/__init__.py`
3. Ensure decorator uses correct string name
4. Check for syntax errors in model file

### Dataset Not Registering

**Problem**: Dataset not found by name

**Solutions**:
1. Verify `_name` class attribute is set
2. Check dataset imported in `src/hyrax/data_sets/__init__.py`
3. Ensure `_name` matches config `[data] name = "..."`
4. Check class actually subclasses `HyraxDataset` or `HyraxImageDataset`

### Verb Not in CLI

**Problem**: Verb command not available

**Solutions**:
1. Verify `@hyrax_verb("verb_name")` decorator present
2. Check verb imported in `src/hyrax/verbs/__init__.py`
3. Ensure `setup_parser()` and `run_cli()` methods exist
4. Check `add_parser_kwargs` is set
5. Verify no syntax errors in verb file

## External Plugins

For external packages:
```toml
[model]
name = "my_package.MyModel"  # Triggers auto-load of my_package/default_config.toml
```

Hyrax will:
1. Import `my_package.MyModel`
2. Look for `my_package/default_config.toml`
3. Merge with local config

## Best Practices

### Models
1. Use descriptive model names
2. Document architecture in docstring
3. Implement all required methods
4. Use `prepare_inputs()`, not deprecated `to_tensor()`
5. Return loss dictionary from `train_step()`
6. Add comprehensive tests

### Datasets
1. Subclass appropriate base (`HyraxDataset` or `HyraxImageDataset`)
2. Use `_name` attribute for registration
3. Return dictionaries from `__getitem__()`
4. Include metadata in returned items
5. Handle splits (train/val/test) appropriately
6. Use Pooch for reproducible data downloads in tests

### Verbs
1. Use clear, action-oriented names
2. Provide helpful CLI help text
3. Implement both `run()` and `run_cli()`
4. Return meaningful exit codes
5. Save results to timestamped directories
6. Print progress and summary information

## Related Skills

- For testing components, see: **Hyrax Testing Strategy**
- For overall workflow, see: **Hyrax Development Workflow**
- For config setup, see: **Hyrax Configuration System**

## References

- Complete guide: [HYRAX_GUIDE.md](../../HYRAX_GUIDE.md#adding-new-components)
- Model registry: `src/hyrax/models/model_registry.py`
- Dataset registry: `src/hyrax/data_sets/data_set_registry.py`
- Verb registry: `src/hyrax/verbs/verb_registry.py`
