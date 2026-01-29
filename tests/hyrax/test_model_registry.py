import logging

from torch import nn

from hyrax import Hyrax
from hyrax.models.model_registry import hyrax_model


def test_use_model_optimizer():
    """Test that the config will not override an optimizer defined in the model."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)
            self.optimizer = "model_optimizer"
            self.scheduler = None

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("optimizer.name", "torch.optim.SGD")

    model = TestModel(h.config)
    assert hasattr(model, "optimizer")
    assert model.optimizer == "model_optimizer"  # Should use the model's own optimizer, not the config


def test_use_config_optimizer():
    """Test that the config will inject an optimizer if the model does not define one."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("optimizer.name", "torch.optim.SGD")

    model = TestModel(h.config)
    assert hasattr(model, "optimizer")
    assert model.optimizer.__class__.__name__ == "SGD"  # Should use the config's optimizer


def test_no_optimizer_defined_logs_warning(caplog):
    """Test that if neither model nor config define an optimizer, a warning is logged."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)
            self.scheduler = None

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("optimizer.name", "")

    with caplog.at_level(logging.WARNING):
        _ = TestModel(h.config)
        assert "No optimizer specified in config or" in caplog.text


def test_use_model_criterion():
    """Test that the config will not override a criterion defined in the model."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)
            self.criterion = "model_criterion"

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("criterion.name", "torch.nn.MSELoss")

    model = TestModel(h.config)
    assert hasattr(model, "criterion")
    assert model.criterion == "model_criterion"  # Should use the model's own criterion, not the config


def test_use_config_criterion():
    """Test that the config will inject a criterion if the model does not define one."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("criterion.name", "torch.nn.MSELoss")

    model = TestModel(h.config)
    assert hasattr(model, "criterion")
    assert model.criterion.__class__.__name__ == "MSELoss"  # Should use the config's criterion


def test_no_criterion_defined_logs_warning(caplog):
    """Test that if neither model nor config define a criterion, a warning is logged."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("criterion.name", "")

    with caplog.at_level(logging.WARNING):
        _ = TestModel(h.config)
        assert "No criterion specified in config or" in caplog.text


def test_criterion_defined_in_model_and_config(caplog):
    """Test that if both model and config define a criterion, a warning is logged."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)
            self.criterion = "model_criterion"

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("criterion.name", "torch.nn.MSELoss")

    with caplog.at_level(logging.WARNING):
        model = TestModel(h.config)
        assert "Both model and config define a criterion" in caplog.text

    assert model.criterion == "model_criterion"  # Should use the model's own criterion


def test_optimizer_defined_in_model_and_config(caplog):
    """Test that if both model and config define an optimizer, a warning is logged."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)
            self.optimizer = "model_optimizer"
            self.scheduler = None

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("optimizer.name", "torch.optim.SGD")

    with caplog.at_level(logging.WARNING):
        model = TestModel(h.config)
        assert "Both model and config define an optimizer" in caplog.text

    assert model.optimizer == "model_optimizer"  # Should use the model's own optimizer


def test_use_model_scheduler():
    """Test that the config will not override a scheduler defined in the model."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)
            self.scheduler = "model_scheduler"

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("scheduler.name", "torch.optim.lr_scheduler.ConstantLR")

    model = TestModel(h.config)
    assert hasattr(model, "scheduler")
    assert model.scheduler == "model_scheduler"  # Should use the model's own scheduler, not the config
  
  
def test_use_config_scheduler(caplog):
    """Test that the config will inject a scheduler if the model does not define one."""
    
    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("scheduler.name", "torch.optim.lr_scheduler.ConstantLR")

    model = TestModel(h.config)
    assert hasattr(model, "scheduler")
    assert model.scheduler.__class__.__name__ == "ConstantLR"  # Should use the config's scheduler
  

def test_no_scheduler_defined_logs_warning(caplog):
    """Test that if neither model nor config define a scheduler, a warning is logged."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("scheduler.name", "")

    with caplog.at_level(logging.WARNING):
        _ = TestModel(h.config)
        assert "No scheduler specified in config or" in caplog.text
  
  
def test_scheduler_defined_in_model_and_config(caplog):
    """Test that if both model and config define a scheduler, a warning is logged."""

    @hyrax_model
    class TestModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)
            self.optimizer = "model_optimizer"
            self.scheduler = "model_scheduler"

    h = Hyrax()
    h.set_config("model.name", "TestModel")
    h.set_config("optimizer.name", "torch.optim.SGD")
    h.set_config("scheduler.name", "torch.optim.lr_scheduler.ConstantLR")

    with caplog.at_level(logging.WARNING):
        model = TestModel(h.config)
        assert "Both model and config define a scheduler" in caplog.text

    assert model.scheduler == "model_scheduler"  # Should use the model's own scheduler