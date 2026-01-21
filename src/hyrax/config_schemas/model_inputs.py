"""
Pydantic models describing the structure of the ``model_inputs`` configuration.

These schemas are passive type definitions only and are not yet wired into the
runtime configuration loading logic.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .base import BaseConfigModel


class ModelInputsConfig(BaseConfigModel):
    """Per-dataset configuration used within ``model_inputs``."""

    dataset_class: str = Field(..., description="Fully qualified dataset class name.")
    data_location: str | None = Field(
        None, description="Path or URI describing where the dataset is stored."
    )
    fields: list[str] | None = Field(
        None, description="Subset of columns/fields to request from the dataset."
    )
    primary_id_field: str | None = Field(
        None, description="Name of the primary identifier field in the dataset."
    )
    dataset_config: dict[str, Any] | None = Field(
        None, description="Dataset-specific configuration to pass through to the class."
    )

    @model_validator(mode="before")
    @classmethod
    def unwrap_data_key(cls, value: Any) -> Any:
        """Allow configurations specified under a ``data`` wrapper."""

        if isinstance(value, dict) and "data" in value and len(value) == 1:
            return value["data"]
        return value

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain dictionary."""

        return self.model_dump()


class ModelInputsDefinition(BaseConfigModel):
    """Typed representation of the full ``model_inputs`` table."""

    train: ModelInputsConfig | None = Field(
        None, description="Dataset configuration used for training."
    )
    validate: ModelInputsConfig | None = Field(
        None, description="Dataset configuration used for validation."
    )
    infer: ModelInputsConfig | None = Field(
        None, description="Dataset configuration used for inference."
    )
    other_datasets: dict[str, ModelInputsConfig] = Field(
        default_factory=dict,
        description="Additional dataset definitions keyed by friendly name.",
    )

    @model_validator(mode="before")
    @classmethod
    def collect_additional_datasets(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Capture arbitrary dataset keys beyond train/validate/infer."""

        known = {"train", "validate", "infer"}
        extra: dict[str, Any] = {}
        for key in list(values.keys()):
            if key not in known:
                extra[key] = values.pop(key)
        values.setdefault("other_datasets", {})
        values["other_datasets"].update(extra)
        return values

    def as_dict(self) -> dict[str, Any]:
        """Export as a nested dictionary compatible with existing configs."""

        output: dict[str, Any] = {}
        for name in ("train", "validate", "infer"):
            value = getattr(self, name)
            if value is not None:
                output[name] = {"data": value.as_dict()}
        for key, cfg in self.other_datasets.items():
            output[key] = {"data": cfg.as_dict()}
        return output
