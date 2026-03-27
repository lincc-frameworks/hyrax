"""
Pydantic models describing the structure of the ``data_request`` configuration.

These schemas validate and enforce the structure of dataset requests used throughout
the Hyrax framework.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
        parsed = urlparse(v)
        if parsed.scheme and v.startswith(f"{parsed.scheme}://"):
            return v
        return str(Path(v).expanduser().resolve())

    @model_validator(mode="after")
    def require_primary_id_for_split_fraction(self) -> DataRequestConfig:
        """Ensure that split_fraction is only set when primary_id_field is also provided."""
        if self.split_fraction is not None and self.primary_id_field is None:
            raise ValueError("'split_fraction' can only be specified when 'primary_id_field' is also set.")
        return self

    def as_dict(self, *, exclude_unset: bool = False) -> dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return self.model_dump(exclude_unset=exclude_unset)


# Type alias for a dataset group value: a dict mapping friendly names to configs.
DatasetGroupValue = dict[str, DataRequestConfig]


def _normalize_dataset_group(value: Any) -> DatasetGroupValue:
    """Normalize a single dataset group value into a ``dict[str, DataRequestConfig]``.

    Every dataset source within a group must be identified by a user-supplied
    *friendly name*.  The friendly name is the key in the returned dict and is
    used by ``DataProvider`` to reference the dataset at runtime.

    Accepted inputs
    ---------------
    - A ``dict`` whose values are ``DataRequestConfig`` instances or plain dicts
      that can be validated as one.  The keys become the friendly names.

    Rejected inputs (raise ``ValueError``)
    ----------------------------------------
    - A flat dict that contains ``dataset_class`` at the top level (no friendly
      name wrapper).
    - A bare ``DataRequestConfig`` instance (no friendly name wrapper).
    """

    if isinstance(value, DataRequestConfig):
        raise ValueError(
            "A friendly name is required for each dataset source. "
            "Wrap the config in a dict with a friendly name, e.g. "
            '{"<friendly_name>": <DataRequestConfig>}.'
        )

    if isinstance(value, dict):
        # Detect a flat config: dataset_class at the top level means no friendly
        # name was provided.
        if "dataset_class" in value:
            raise ValueError(
                "A friendly name is required for each dataset source. "
                'Instead of {"dataset_class": ..., ...}, use '
                '{"<friendly_name>": {"dataset_class": ..., ...}}.'
            )

        # Dict of named configs — parse each value.
        parsed_dict: DatasetGroupValue = {}
        for key, val in value.items():
            if isinstance(val, DataRequestConfig):
                parsed_dict[key] = val
            elif isinstance(val, dict):
                parsed_dict[key] = DataRequestConfig.model_validate(val)
            else:
                raise ValueError(
                    f"Value for friendly name '{key}' must be a dict or DataRequestConfig instance, "
                    f"got {type(val).__name__}."
                )
        return parsed_dict

    raise ValueError(f"Cannot parse dataset group value of type {type(value).__name__}")


def _iter_all_configs(
    groups: dict[str, DatasetGroupValue],
) -> list[tuple[str, DataRequestConfig]]:
    """Yield ``(group_name, config)`` pairs across all groups."""
    result = []
    for group_name, group_value in groups.items():
        for config in group_value.values():
            result.append((group_name, config))
    return result


class DataRequestDefinition(RootModel[dict[str, DatasetGroupValue]]):
    """Typed representation of the full ``data_request`` table.

    Accepts any number of arbitrarily-named dataset groups (e.g. ``train``,
    ``validate``, ``infer``, ``test``, ``finetune``, …).  Each group value is
    a ``dict`` of *friendly-named* ``DataRequestConfig`` instances.  A friendly
    name must always be provided explicitly — the schema will raise a validation
    error if a dataset source is specified without one.

    Example (Python)::

        {
            "train": {
                "my_dataset": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "/path/to/data",
                    "primary_id_field": "object_id",
                }
            }
        }

    Example (TOML)::

        [data_request.train.my_dataset]
        dataset_class = "HyraxRandomDataset"
        data_location = "/path/to/data"
        primary_id_field = "object_id"
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
            primary_count = sum(1 for config in group_value.values() if config.primary_id_field is not None)

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

    def validate_cross_group(self, groups: set[str]) -> None:
        """Run cross-group split_fraction checks restricted to the specified groups.

        This method is intended to be called by verb classes at instantiation time,
        scoped to only the dataset groups the verb actually uses (via
        ``REQUIRED_DATA_GROUPS`` and ``OPTIONAL_DATA_GROUPS``).  By restricting
        validation to active groups, configs that contain groups irrelevant to the
        current verb do not cause false validation failures.

        Parameters
        ----------
        groups : set[str]
            Set of active group names to validate.  Only configs belonging to
            these groups are considered.

        Raises
        ------
        ValueError
            If split_fraction values for a given ``data_location`` sum to more
            than 1.0, or if split_fraction consistency is violated (some configs
            for a location set it while others do not).
        """
        filtered = {k: v for k, v in self.root.items() if k in groups}

        # Check that split_fraction values for the same data_location do not exceed 1.0.
        fractions_by_location: dict[str, list[float]] = defaultdict(list)
        for _group_name, config in _iter_all_configs(filtered):
            if config.split_fraction is not None:
                fractions_by_location[config.data_location].append(config.split_fraction)

        for location, fractions in fractions_by_location.items():
            total = sum(fractions)
            if np.round(total, decimals=5) > 1.0:
                raise ValueError(
                    f"The sum of split_fraction values for data_location '{location}' "
                    f"is {total}, which exceeds 1.0."
                )

        # Check that all configs sharing a data_location either all set split_fraction or none do.
        configs_by_location: dict[str, list[tuple[str, DataRequestConfig]]] = defaultdict(list)
        for group_name, config in _iter_all_configs(filtered):
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

    def __contains__(self, key: str) -> bool:
        """Return True if the group name is present in the definition."""
        return key in self.root

    def __getitem__(self, key: str) -> DatasetGroupValue:
        """Return the dataset group value for the given group name."""
        return self.root[key]

    def as_dict(self, *, exclude_unset: bool = False) -> dict[str, Any]:
        """Export as a nested dictionary compatible with existing configs.

        Each group value is a dict of ``{friendly_name: flat_config_dict}``.
        No implicit ``"data"`` wrapper is added — the friendly names supplied
        by the user are preserved verbatim.
        """
        return {
            name: {key: cfg.as_dict(exclude_unset=exclude_unset) for key, cfg in value.items()}
            for name, value in self.root.items()
        }
