"""
Typed configuration schemas for Hyrax.

This package will house Pydantic models that describe and validate Hyrax
configuration files.  For now it exposes the base schema stub to allow
incremental adoption in downstream modules and tests.
"""

from .base import BaseConfigModel
from .data_request import DataRequestConfig, DataRequestDefinition
from .datasets import (
    DownloadedLSSTDatasetConfig,
    HSCDataSetConfig,
    HyraxCifarDatasetConfig,
    HyraxCSVDatasetConfig,
    HyraxRandomDatasetConfig,
    LSSTDatasetConfig,
)

__all__ = [
    "BaseConfigModel",
    "DataRequestConfig",
    "DataRequestDefinition",
    "HyraxRandomDatasetConfig",
    "HyraxCifarDatasetConfig",
    "LSSTDatasetConfig",
    "DownloadedLSSTDatasetConfig",
    "HSCDataSetConfig",
    "HyraxCSVDatasetConfig",
]
