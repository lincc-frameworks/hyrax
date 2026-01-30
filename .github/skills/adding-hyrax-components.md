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
1. Subclass `HyraxDataset` directly or indirectly through child classes
2. Set `_name` class attribute (triggers auto-registration)
3. Implement: `__len__()`, `__getitem__()`, metadata interface

### Step-by-Step Process

#### 1. Create Dataset File

Location: `src/hyrax/data_sets/my_dataset.py`

```python
from pathlib import Path
from hyrax.data_sets.data_set_registry import HyraxDataset

class MyDataset(HyraxDataset):
    """Custom dataset for my astronomy data.
    
    This dataset loads tabular data from <describe data source>.
    """
    
    # Auto-registration via _name attribute
    _name = "MyDataset"
    
    def __init__(self, config, data_location=None):
        """Initialize dataset.
        
        Args:
            config: Hyrax config dictionary
            data_location: Path to data file
        """
        self.data_location = data_location or Path(config["data"]["data_location"])
        
        # Load your data here
        self.data = self._load_data()
        
        # Call parent __init__ at the end
        super().__init__(config)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get single data item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with data item
        """
        return {
            "data": self.data[idx],
            "label": self._get_label(idx) if hasattr(self, '_get_label') else None,
        }
    
    def _load_data(self):
        """Load data from file."""
        # Implementation depends on data format
        # e.g., np.load(), pd.read_csv(), etc.
        pass
    
    def sample_data(self):
        """Return the first record as a sample."""
        return {"data": self.data[0]}
    
    @classmethod
    def is_map(cls):
        """Indicate this is a map-style dataset."""
        return True
```

#### 2. Registration (Automatic)

Setting `_name = "MyDataset"` triggers automatic registration via `__init_subclass__`.

#### 3. Add Default Config

Location: `src/hyrax/data_sets/my_dataset_default_config.toml`

```toml
[data]
name = "MyDataset"
data_location = "/path/to/data"
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
    config = {"data": {"name": "MyDataset", "data_location": str(tmp_path / "data.csv")}}
    dataset = MyDataset(config, data_location=tmp_path / "data.csv")
    assert dataset is not None

def test_my_dataset_length(tmp_path):
    """Test dataset length."""
    # Setup test data
    config = {"data": {"name": "MyDataset", "data_location": str(tmp_path / "data.csv")}}
    dataset = MyDataset(config, data_location=tmp_path / "data.csv")
    assert len(dataset) >= 0

def test_my_dataset_getitem(tmp_path):
    """Test getting data item."""
    config = {"data": {"name": "MyDataset", "data_location": str(tmp_path / "data.csv")}}
    dataset = MyDataset(config, data_location=tmp_path / "data.csv")
    if len(dataset) > 0:
        item = dataset[0]
        assert "data" in item
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
from hyrax.verbs.verb_registry import hyrax_verb, Verb
from hyrax import Hyrax

@hyrax_verb
class MyVerb(Verb):
    """Custom verb for doing something useful.
    
    This verb performs <describe functionality>.
    """
    
    # CLI name and help text
    cli_name = "myverb"
    add_parser_kwargs = {
        "help": "Perform my custom operation",
        "description": "Detailed description of what this verb does."
    }
    
    @staticmethod
    def setup_parser(parser):
        """Setup argument parser for CLI.
        
        Most verbs don't need CLI arguments - prefer configuration over parameters.
        """
        pass
    
    def run(self):
        """Execute verb logic.
        
        Returns:
            Results dictionary
        """
        # Access config from self.config
        config = self.config
        
        # Initialize Hyrax if needed
        # hyrax = Hyrax(config)
        
        # Perform verb logic
        results = self._perform_operation()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def run_cli(self, args=None):
        """Execute verb from CLI.
        
        Args:
            args: Parsed command line arguments (usually None)
            
        Returns:
            Exit code (0 for success)
        """
        # Run the verb
        results = self.run()
        
        # Print summary
        print(f"Operation complete: {results.get('summary', 'Done')}")
        
        return 0
    
    def _perform_operation(self):
        """Internal method to perform the actual work."""
        # Implementation here
        return {"summary": "Success"}
    
    def _save_results(self, results):
        """Save results to output directory."""
        # Implementation here
        pass
```

#### 2. Registration (Automatic)

The `@hyrax_verb` decorator handles automatic CLI registration.

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
from hyrax import Hyrax

def test_my_verb_run(default_config):
    """Test verb execution."""
    hyrax = Hyrax(default_config)
    verb = MyVerb(hyrax)
    results = verb.run()
    assert results is not None
    assert "summary" in results
```

#### 5. Use via CLI

```bash
# View help
hyrax myverb --help

# Execute verb (config loaded from hyrax CLI framework)
hyrax myverb -c config.toml
```

## Best Practices

### Models
1. Use descriptive model names
2. Document architecture in docstring
3. Implement all required methods
4. Use `prepare_inputs()`, not deprecated `to_tensor()`
5. Return loss dictionary from `train_step()`
6. Add comprehensive tests

### Datasets
1. Subclass `HyraxDataset` directly or indirectly
2. Use `_name` attribute for registration
3. Return dictionaries from `__getitem__()`
4. Implement the metadata interface
5. Call `super().__init__(config)` at the end of your `__init__`
6. Use Pooch for reproducible data downloads in tests

### Verbs
1. Use clear, action-oriented names
2. Provide helpful CLI help text in `add_parser_kwargs`
3. Implement both `run()` and `run_cli()`
4. Prefer configuration over CLI parameters
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
