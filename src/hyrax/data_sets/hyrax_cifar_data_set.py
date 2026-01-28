"""Compatibility shim for legacy import path.

Sphinx/autoapi references ``hyrax.data_sets.hyrax_cifar_data_set``; this module
re-exports the current implementation from ``hyrax_cifar_dataset``.
"""

from hyrax.data_sets.hyrax_cifar_dataset import HyraxCifarDataset

# For backward compatibility
HyraxCifarDataSet = HyraxCifarDataset

__all__ = ["HyraxCifarDataset", "HyraxCifarDataSet"]
