"""
Base classes for typed Hyrax configuration.

This module introduces a minimal Pydantic model that future configuration
schemas will inherit from. It intentionally contains no runtime logic or
validation rules beyond what the Pydantic `BaseModel` provides by default.
"""

from pydantic import BaseModel


class BaseConfigModel(BaseModel):
    """Base class for future Hyrax configuration schemas."""

    class Config:
        # Align with current behavior: allow population from names as-is.
        populate_by_name = True
