"""Dataset-specific configuration schemas."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import BaseConfigModel


class HyraxRandomDatasetConfig(BaseConfigModel):
    """Configuration for :class:`HyraxRandomDataset`."""

    size: int = Field(..., description="Number of random samples to generate.")
    shape: list[int] = Field(..., description="Shape of each random sample.")
    seed: int = Field(..., description="Random seed for reproducibility.")
    provided_labels: list[int] | list[str] | bool = Field(
        False, description="Optional labels to sample from; set to false for none."
    )
    metadata_fields: list[str] | bool = Field(
        False, description="Metadata field names to include; false for none."
    )
    number_invalid_values: int = Field(
        0, description="Count of invalid values to inject into the random data."
    )
    invalid_value_type: str | float = Field(
        "nan",
        description='Type of invalid value to insert; one of "nan", "inf", "-inf", "none" or a float.',
    )


class HyraxCifarDatasetConfig(BaseConfigModel):
    """Configuration for :class:`HyraxCifarDataset`."""

    use_training_data: bool = Field(
        True, description="If true, download CIFAR10 training split; otherwise use test split."
    )


class LSSTDatasetConfig(BaseConfigModel):
    """Configuration for :class:`LSSTDataset`."""

    filters: list[str] | bool = Field(
        False, description="Bands to include; false to include all default bands."
    )
    hats_catalog: str | None = Field(
        None, description="Path to HATS catalog. Required if astropy_table is not provided."
    )
    astropy_table: str | None = Field(
        None, description="Path to an Astropy-readable table. Required if hats_catalog is not provided."
    )
    semi_width_deg: float = Field(..., description="Semi width of cutouts in degrees.")
    semi_height_deg: float = Field(..., description="Semi height of cutouts in degrees.")
    object_id_column_name: str | bool = Field(
        False, description="Override object ID column name; false to autodetect."
    )
    butler_repo: str | None = Field(None, description="Butler repository path.")
    butler_collection: str | None = Field(None, description="Butler collection name.")
    skymap: str | None = Field(None, description="Butler skymap name.")


class DownloadedLSSTDatasetConfig(LSSTDatasetConfig):
    """Configuration for :class:`DownloadedLSSTDataset`."""

    # Inherits LSSTDatasetConfig fields
    pass


class HSCDataSetConfig(BaseConfigModel):
    """Configuration for :class:`HSCDataSet`."""

    filters: list[str] | bool = Field(False, description="Filters to include; false for all.")
    filter_catalog: str | bool = Field(
        False, description="Path to filter catalog; false to disable."
    )
    object_id_column_name: str | bool = Field(
        False, description="Override object ID column name; false for default."
    )
    filter_column_name: str | bool = Field(
        False, description="Override filter column name; false for default."
    )
    filename_column_name: str | bool = Field(
        False, description="Override filename column name; false for default."
    )


class HyraxCSVDatasetConfig(BaseConfigModel):
    """Configuration for :class:`HyraxCSVDataset`."""

    # Currently no dataset-specific fields beyond data_location/fields in ModelInputsConfig.
    extra: dict[str, Any] | None = Field(
        None, description="Placeholder for future CSV dataset options."
    )
