"""
Typed configuration schemas for Hyrax.

This package will house Pydantic models that describe and validate Hyrax
configuration files.  For now it exposes the base schema stub to allow
incremental adoption in downstream modules and tests.
"""

from .base import BaseConfigModel
from .model_inputs import ModelInputsConfig, ModelInputsDefinition

__all__ = ["BaseConfigModel", "ModelInputsConfig", "ModelInputsDefinition"]
