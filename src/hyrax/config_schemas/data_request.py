"""
Pydantic models describing the structure of the ``data_request`` configuration.

These schemas validate and enforce the structure of dataset requests used throughout
the Hyrax framework.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, RootModel, field_validator, model_validator

from .base import BaseConfigModel


class DataRequestConfig(BaseConfigModel):
    """Per-dataset configuration used within ``data_request``."""

    dataset_class: str = Field(..., description="Fully qualified dataset class name.")
    data_location: str = Field(..., description="Path or URI describing where the dataset is stored.")
    fields: list[str] | None = Field(
        None, description="Subset of columns/fields to request from the dataset."
    )
    primary_id_field: str | None = Field(
        None, description="Name of the primary identifier field in the dataset."
    )

    split_fraction: float | None = Field(
        None,
        description="Fraction of the dataset to use, must be greater than 0.0 and at most 1.0.",
        gt=0.0,
        le=1.0,
    )

    dataset_config: dict | None = Field(
        None,
        description="Dataset-specific configuration as a free-form dictionary.",
    )

    @field_validator("data_location")
    @classmethod
    def resolve_data_location(cls, v: str) -> str:
        """Fully resolve the data_location path, expanding user home directories
        and converting relative paths to absolute paths."""
        return str(Path(v).expanduser().resolve())

    @model_validator(mode="after")
    def require_primary_id_for_split_fraction(self) -> DataRequestConfig:
        """Ensure that split_fraction is only set when primary_id_field is also provided."""
        if self.split_fraction is not None and self.primary_id_field is None:
            raise ValueError("'split_fraction' can only be specified when 'primary_id_field' is also set.")
        return self

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


# Type alias for a dataset group value: either a single config or a dict of named configs.
DatasetGroupValue = DataRequestConfig | dict[str, DataRequestConfig]


def _normalize_dataset_group(value: Any) -> DatasetGroupValue:
    """Normalize a single dataset group value into a ``DataRequestConfig``
    or a ``dict[str, DataRequestConfig]``.

    Handles:
    - A ``DataRequestConfig`` instance (returned as-is)
    - A dict that looks like a single config (has ``dataset_class`` or ``data`` key)
    - A dict of multiple configs (values are dicts with ``dataset_class``)
    """

    if isinstance(value, DataRequestConfig):
        return value

    if isinstance(value, dict):
        # Check if this looks like a single config (has dataset_class)
        # or a dict of configs (values are dicts with dataset_class)
        if "dataset_class" in value or "data" in value:
            # Single config
            return DataRequestConfig.model_validate(value)
        else:
            # Dict of configs - parse each value
            parsed_dict = {}
            for key, val in value.items():
                if isinstance(val, DataRequestConfig):
                    parsed_dict[key] = val
                else:
                    parsed_dict[key] = DataRequestConfig.model_validate(val)
            return parsed_dict

    raise ValueError(f"Cannot parse dataset group value of type {type(value).__name__}")


def _iter_all_configs(
    groups: dict[str, DatasetGroupValue],
) -> list[tuple[str, DataRequestConfig]]:
    """Yield ``(group_name, config)`` pairs across all groups."""
    result = []
    for group_name, group_value in groups.items():
        configs_dict = group_value if isinstance(group_value, dict) else {"_default": group_value}
        for config in configs_dict.values():
            result.append((group_name, config))
    return result


class DataRequestDefinition(RootModel[dict[str, DatasetGroupValue]]):
    """Typed representation of the full ``data_request`` table.

    Accepts any number of arbitrarily-named dataset groups (e.g. ``train``,
    ``validate``, ``infer``, ``test``, ``finetune``, …).  Each group value is
    either a single ``DataRequestConfig`` or a ``dict`` of named
    ``DataRequestConfig`` instances.
    """

    @model_validator(mode="before")
    @classmethod
    def normalize_all_groups(cls, value: Any) -> dict[str, DatasetGroupValue]:
        """Parse every top-level key into the expected group format."""
        if not isinstance(value, dict):
            raise ValueError("DataRequestDefinition expects a dictionary of dataset groups.")

        normalized: dict[str, DatasetGroupValue] = {}
        for group_name, group_value in value.items():
            if group_value is None:
                continue
            normalized[group_name] = _normalize_dataset_group(group_value)

        return normalized

    @model_validator(mode="after")
    def require_at_least_one_dataset(self) -> DataRequestDefinition:
        """Ensure at least one dataset group is provided."""
        if not self.root:
            raise ValueError("At least one dataset group must be provided.")
        return self

    @model_validator(mode="after")
    def validate_primary_id_fields(self) -> DataRequestDefinition:
        """Validate that exactly one DataRequestConfig in each dataset group
        has a non-None primary_id_field.

        This ensures that when multiple datasets are requested (e.g., a group
        contains a dict of multiple DataRequestConfig instances), exactly
        one of them specifies which field to use as the primary identifier.
        """
        for group_name, group_value in self.root.items():
            configs_dict = group_value if isinstance(group_value, dict) else {"_default": group_value}

            primary_count = sum(1 for config in configs_dict.values() if config.primary_id_field is not None)

            if primary_count == 0:
                raise ValueError(
                    f"'{group_name}' must have exactly one DataRequestConfig with "
                    f"'primary_id_field' set, but found none."
                )
            elif primary_count > 1:
                raise ValueError(
                    f"'{group_name}' must have exactly one DataRequestConfig with "
                    f"'primary_id_field' set, but found {primary_count}."
                )

        return self

    @model_validator(mode="after")
    def validate_split_fraction_sums(self) -> DataRequestDefinition:
        """Validate that the sum of split_fraction values for configs sharing
        the same data_location does not exceed 1.0.

        This check spans across all dataset groups to ensure that the total
        fraction requested from a single data source is not more than 100%.
        """
        fractions_by_location: dict[str, list[float]] = defaultdict(list)

        for _group_name, config in _iter_all_configs(self.root):
            if config.split_fraction is not None:
                fractions_by_location[config.data_location].append(config.split_fraction)

        for location, fractions in fractions_by_location.items():
            total = sum(fractions)
            if np.round(total, decimals=5) > 1.0:
                raise ValueError(
                    f"The sum of split_fraction values for data_location '{location}' "
                    f"is {total}, which exceeds 1.0."
                )

        return self

    @model_validator(mode="after")
    def validate_split_fraction_consistency(self) -> DataRequestDefinition:
        """Validate that if any config specifies a split_fraction for a given
        data_location, then all other configs sharing that data_location must
        also specify a split_fraction.

        This prevents ambiguous situations where some configs claim a fraction
        of a dataset while others implicitly claim the remainder or the whole.
        """
        # Group all configs by data_location
        configs_by_location: dict[str, list[tuple[str, DataRequestConfig]]] = defaultdict(list)

        for group_name, config in _iter_all_configs(self.root):
            configs_by_location[config.data_location].append((group_name, config))

        for location, group_configs in configs_by_location.items():
            if len(group_configs) < 2:
                continue

            has_fraction = [cfg.split_fraction is not None for _, cfg in group_configs]

            if any(has_fraction) and not all(has_fraction):
                missing_groups = [
                    group_name for (group_name, cfg) in group_configs if cfg.split_fraction is None
                ]
                raise ValueError(
                    f"All configs sharing data_location '{location}' must specify "
                    f"'split_fraction' when any of them does. Missing in: "
                    f"{', '.join(missing_groups)}."
                )

        return self

    def __contains__(self, key: str) -> bool:
        return key in self.root

    def __getitem__(self, key: str) -> DatasetGroupValue:
        return self.root[key]

    def as_dict(self, *, exclude_unset: bool = False) -> dict[str, Any]:
        """Export as a nested dictionary compatible with existing configs."""
        output: dict[str, Any] = {}

        for name, value in self.root.items():
            if isinstance(value, dict):
                output[name] = {
                    key: {"data": cfg.as_dict(exclude_unset=exclude_unset)} for key, cfg in value.items()
                }
            else:
                output[name] = {"data": value.as_dict(exclude_unset=exclude_unset)}

        return output
