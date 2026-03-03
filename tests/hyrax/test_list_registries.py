from torch import nn

from hyrax import Hyrax
from hyrax.data_sets.data_set_registry import HyraxDataset
from hyrax.models.model_registry import hyrax_model


def test_list_models_prints_sorted(capsys):
    """list_models() should print model names in alphabetical order."""
    h = Hyrax()
    h.list_models()
    captured = capsys.readouterr()
    names = captured.out.strip().splitlines()
    assert names == sorted(names)
    # At least some built-in models should be present
    assert len(names) > 0


def test_list_models_includes_registered_model(capsys):
    """list_models() should include a newly registered model."""

    @hyrax_model
    class ZZZTestListModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()

    h = Hyrax()
    h.list_models()
    captured = capsys.readouterr()
    assert "ZZZTestListModel" in captured.out


def test_list_dataset_classes_prints_sorted(capsys):
    """list_dataset_classes() should print dataset class names in alphabetical order."""
    h = Hyrax()
    h.list_dataset_classes()
    captured = capsys.readouterr()
    names = captured.out.strip().splitlines()
    assert names == sorted(names)
    assert len(names) > 0


def test_list_dataset_classes_includes_registered_dataset(capsys):
    """list_dataset_classes() should include a newly registered dataset class."""
    from torch.utils.data import Dataset

    class ZZZTestListDataset(HyraxDataset, Dataset):
        def __init__(self, config):
            super().__init__(config)

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return {}

    h = Hyrax()
    h.list_dataset_classes()
    captured = capsys.readouterr()
    assert "ZZZTestListDataset" in captured.out
