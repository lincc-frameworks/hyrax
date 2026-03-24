from torch import nn

from hyrax import Hyrax
from hyrax.datasets.dataset_registry import HyraxDataset
from hyrax.models.model_registry import hyrax_model


def test_list_models_returns_sorted():
    """list_models() should return model names in alphabetical order."""
    h = Hyrax()
    names = h.list_models()
    assert names == sorted(names)
    # At least some built-in models should be present
    assert len(names) > 0


def test_list_models_includes_registered_model():
    """list_models() should include a newly registered model."""

    @hyrax_model
    class ZZZTestListModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config

        def train_batch(self, batch):
            pass

        def infer_batch(self, batch):
            pass

    h = Hyrax()
    assert "ZZZTestListModel" in h.list_models()


def test_list_dataset_classes_returns_sorted():
    """list_dataset_classes() should return dataset class names in alphabetical order."""
    h = Hyrax()
    names = h.list_dataset_classes()
    assert names == sorted(names)
    assert len(names) > 0


def test_list_dataset_classes_includes_registered_dataset():
    """list_dataset_classes() should include a newly registered dataset class."""

    class ZZZTestListDataset(HyraxDataset):
        def __init__(self, config):
            super().__init__(config)

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return {}

    h = Hyrax()
    assert "ZZZTestListDataset" in h.list_dataset_classes()
