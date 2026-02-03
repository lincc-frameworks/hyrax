"""
Pydantic models describing the structure of the ``data_request`` configuration.

These schemas validate and enforce the structure of dataset requests used throughout
the Hyrax framework.
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base import BaseConfigModel


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

    # Keep dataset_config as Any - it's a free-form dictionary for dataset-specific settings
    dataset_config: Any = Field(
        None,
        description="Dataset-specific configuration as a free-form dictionary.",
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

    def as_dict(self, *, exclude_unset: bool = False) -> dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return self.model_dump(exclude_unset=exclude_unset)


class DataRequestDefinition(BaseConfigModel):
    """Typed representation of the full ``data_request`` table."""

    model_config = ConfigDict(protected_namespaces=())

    train: DataRequestConfig | dict[str, DataRequestConfig] | None = Field(
        None, description="Dataset configuration(s) used for training."
    )
    validate: DataRequestConfig | dict[str, DataRequestConfig] | None = Field(
        None, description="Dataset configuration(s) used for validation."
    )
    infer: DataRequestConfig | dict[str, DataRequestConfig] | None = Field(
        None, description="Dataset configuration(s) used for inference."
    )
    other_datasets: dict[str, DataRequestConfig | dict[str, DataRequestConfig]] = Field(
        default_factory=dict,
        description="Additional dataset definitions keyed by friendly name.",
    )

    @field_validator("train", "validate", "infer", mode="before")
    @classmethod
    def normalize_dataset_configs(cls, value: Any) -> Any:
        """
        Normalize dataset config input.

        Handles:
        - Single DataRequestConfig (or dict that can become one)
        - Dict of multiple DataRequestConfig instances
        - Wrapped in {"data": ...} format (already handled by DataRequestConfig)
        """

        if value is None:
            return None

        # If it's already a DataRequestConfig, return as-is
        if isinstance(value, DataRequestConfig):
            return value

        # If it's a dict, check if it's a single config or a dict of configs
        if isinstance(value, dict):
            # Check if this looks like a single config (has dataset_class)
            # or a dict of configs (values are dicts with dataset_class)
            if "dataset_class" in value or "data" in value:
                # Single config - will be parsed as DataRequestConfig
                return value
            else:
                # Appears to be dict of configs - parse each value
                parsed_dict = {}
                for key, val in value.items():
                    if isinstance(val, DataRequestConfig):
                        parsed_dict[key] = val
                    else:
                        parsed_dict[key] = DataRequestConfig.model_validate(val)
                return parsed_dict

        return value

    @model_validator(mode="before")
    @classmethod
    def collect_additional_datasets(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Capture arbitrary dataset keys beyond train/validate/infer."""

        # Copy to avoid mutating the caller's dict
        values = dict(values)
        known = {"train", "validate", "infer"}
        extra = {k: v for k, v in values.items() if k not in known}
        for key in extra:
            values.pop(key)
        values.setdefault("other_datasets", {})
        values["other_datasets"].update(extra)
        return values

    @model_validator(mode="after")
    def validate_primary_id_fields(self) -> DataRequestDefinition:
        """
        Validate that exactly one DataRequestConfig in each dataset group
        has a non-None primary_id_field.

        This ensures that when multiple datasets are requested (e.g., train
        contains a dict of multiple DataRequestConfig instances), exactly
        one of them specifies which field to use as the primary identifier.
        """

        # Check each field that can contain dataset configs
        for field_name in ("train", "validate", "infer"):
            field_value = getattr(self, field_name)

            if field_value is None:
                continue

            # Normalize to dict for uniform processing
            configs_dict = field_value if isinstance(field_value, dict) else {"_default": field_value}

            # Count how many configs have primary_id_field set
            primary_count = sum(1 for config in configs_dict.values() if config.primary_id_field is not None)

            # Validate exactly one
            if primary_count == 0:
                raise ValueError(
                    f"'{field_name}' must have exactly one DataRequestConfig with "
                    f"'primary_id_field' set, but found none."
                )
            elif primary_count > 1:
                raise ValueError(
                    f"'{field_name}' must have exactly one DataRequestConfig with "
                    f"'primary_id_field' set, but found {primary_count}."
                )

        # Also validate other_datasets if present
        if self.other_datasets:
            for dataset_name, config_or_dict in self.other_datasets.items():
                configs_dict = (
                    config_or_dict if isinstance(config_or_dict, dict) else {"_default": config_or_dict}
                )

                primary_count = sum(
                    1 for config in configs_dict.values() if config.primary_id_field is not None
                )

                if primary_count == 0:
                    raise ValueError(
                        f"Dataset '{dataset_name}' must have exactly one DataRequestConfig with "
                        f"'primary_id_field' set, but found none."
                    )
                elif primary_count > 1:
                    raise ValueError(
                        f"Dataset '{dataset_name}' must have exactly one DataRequestConfig with "
                        f"'primary_id_field' set, but found {primary_count}."
                    )

        return self

    def as_dict(self, *, exclude_unset: bool = False) -> dict[str, Any]:
        """Export as a nested dictionary compatible with existing configs."""

        output: dict[str, Any] = {}

        for name in ("train", "validate", "infer"):
            value = getattr(self, name)
            if value is not None:
                # Handle both single config and dict of configs
                if isinstance(value, dict):
                    # Dict of configs - wrap each in {"data": ...}
                    output[name] = {
                        key: {"data": cfg.as_dict(exclude_unset=exclude_unset)} for key, cfg in value.items()
                    }
                else:
                    # Single config - wrap in {"data": ...}
                    output[name] = {"data": value.as_dict(exclude_unset=exclude_unset)}

        for key, cfg in self.other_datasets.items():
            if isinstance(cfg, dict):
                output[key] = {k: {"data": c.as_dict(exclude_unset=exclude_unset)} for k, c in cfg.items()}
            else:
                output[key] = {"data": cfg.as_dict(exclude_unset=exclude_unset)}

        return output
