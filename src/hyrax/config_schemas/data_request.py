"""
Pydantic models describing the structure of the ``data_request`` configuration.

These schemas validate and enforce the structure of dataset requests used throughout
the Hyrax framework.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field, model_validator

from .base import BaseConfigModel
from .datasets import (
    DownloadedLSSTDatasetConfig,
    HSCDataSetConfig,
    HyraxCifarDatasetConfig,
    HyraxCSVDatasetConfig,
    HyraxRandomDatasetConfig,
    LSSTDatasetConfig,
)


class DataRequestConfig(BaseConfigModel):
    """Per-dataset configuration used within ``data_request``."""

    dataset_class: str = Field(..., description="Fully qualified dataset class name.")
    data_location: str | None = Field(None, description="Path or URI describing where the dataset is stored.")
    fields: list[str] | None = Field(
        None, description="Subset of columns/fields to request from the dataset."
    )
    primary_id_field: str | None = Field(
        None, description="Name of the primary identifier field in the dataset."
    )
    _DATASET_SCHEMAS: ClassVar = (
        HyraxRandomDatasetConfig,
        HyraxCifarDatasetConfig,
        LSSTDatasetConfig,
        DownloadedLSSTDatasetConfig,
        HSCDataSetConfig,
        HyraxCSVDatasetConfig,
    )

    DatasetConfigType: ClassVar = (
        HyraxRandomDatasetConfig
        | HyraxCifarDatasetConfig
        | LSSTDatasetConfig
        | DownloadedLSSTDatasetConfig
        | HSCDataSetConfig
        | HyraxCSVDatasetConfig
    )

    dataset_config: DatasetConfigType | dict[str, Any] | None = Field(
        None,
        description=(
            "Dataset-specific configuration. If the dataset_class is a known built-in dataset, "
            "the schema will be validated against its typed config; otherwise a free-form dictionary."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def unwrap_data_key(cls, value: Any) -> Any:
        """Allow configurations specified under a ``data`` wrapper."""

        if (
            isinstance(value, dict)
            and "data" in value
            and len(value) == 1
            and isinstance(value["data"], dict)
        ):
            return value["data"]
        return value

    @model_validator(mode="before")
    @classmethod
    def coerce_dataset_config(cls, value: Any) -> Any:
        """Coerce dataset_config into the appropriate typed model based on dataset_class."""

        if not isinstance(value, dict):
            return value

        dataset_class = value.get("dataset_class")
        cfg = value.get("dataset_config")

        mapping: dict[str, type[BaseConfigModel]] = {
            schema.__name__.removesuffix("Config"): schema for schema in cls._DATASET_SCHEMAS
        }
        # Iterable variants share the same schema
        mapping["HyraxRandomIterableDataset"] = HyraxRandomDatasetConfig
        mapping["HyraxCifarIterableDataset"] = HyraxCifarDatasetConfig

        cfg_model = mapping.get(dataset_class)
        if cfg is not None and cfg_model is not None and not isinstance(cfg, cfg_model):
            value["dataset_config"] = cfg_model.model_validate(cfg)

        return value

    def as_dict(self, *, exclude_unset: bool = False) -> dict[str, Any]:
        """Return the configuration as a plain dictionary."""

        return self.model_dump(exclude_unset=exclude_unset)


class DataRequestDefinition(BaseConfigModel):
    """Typed representation of the full ``data_request`` table."""

    train: DataRequestConfig | None = Field(None, description="Dataset configuration used for training.")
    validate: DataRequestConfig | None = Field(None, description="Dataset configuration used for validation.")
    infer: DataRequestConfig | None = Field(None, description="Dataset configuration used for inference.")
    other_datasets: dict[str, DataRequestConfig] = Field(
        default_factory=dict,
        description="Additional dataset definitions keyed by friendly name.",
    )

    @model_validator(mode="before")
    @classmethod
    def collect_additional_datasets(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Capture arbitrary dataset keys beyond train/validate/infer."""

        known = {"train", "validate", "infer"}
        extra = {k: v for k, v in values.items() if k not in known}
        for key in extra:
            values.pop(key)
        values.setdefault("other_datasets", {})
        values["other_datasets"].update(extra)
        return values

    def as_dict(self, *, exclude_unset: bool = False) -> dict[str, Any]:
        """Export as a nested dictionary compatible with existing configs."""

        output: dict[str, Any] = {}
        for name in ("train", "validate", "infer"):
            value = getattr(self, name)
            if value is not None:
                output[name] = {"data": value.as_dict(exclude_unset=exclude_unset)}
        for key, cfg in self.other_datasets.items():
            output[key] = {"data": cfg.as_dict(exclude_unset=exclude_unset)}
        return output
