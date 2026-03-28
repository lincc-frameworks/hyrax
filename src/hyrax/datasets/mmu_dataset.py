import itertools
import re
from pathlib import Path
from types import MethodType
from typing import Any

from hyrax.datasets.dataset_registry import HyraxDataset


class _IndexedSubset:
    """Fallback wrapper to enforce a max row count for indexable datasets."""

    def __init__(self, dataset: Any, max_samples: int):
        self._dataset = dataset
        self._max_samples = max_samples

    def __getitem__(self, idx: int) -> Any:
        if idx >= self._max_samples:
            raise IndexError(idx)
        return self._dataset[idx]

    def __len__(self) -> int:
        return self._max_samples


class MultimodalUniverseDataset(HyraxDataset):
    """Load a MultimodalUniverse dataset through Hugging Face ``datasets``.

    This dataset class is intentionally generic so one configuration pattern can
    be used for image, spectra, and time-series MMU datasets.

    Examples
    --------
    Example ``data_request`` configuration::

        {
            "infer": {
                "mmu": {
                    "dataset_class": "MultimodalUniverseDataset",
                    "data_location": "hf://MultimodalUniverse/plasticc",
                    "primary_id_field": "object_id",
                    "dataset_config": {
                        "MultimodalUniverseDataset": {
                            "split": "train",
                            "max_samples": 32,
                        }
                    },
                }
            }
        }
    """

    def __init__(self, config: dict, data_location: Path | str | None = None):
        if data_location is None:
            raise ValueError(
                "A `data_location` must be provided. Use either a local dataset path or "
                "a Hugging Face URI like 'hf://MultimodalUniverse/plasticc'."
            )

        self.data_location = str(data_location)
        self.dataset_settings = (
            config.get("data_set", {}).get("MultimodalUniverseDataset", {}) if config is not None else {}
        )

        self.split = self.dataset_settings.get("split", "train")
        self.max_samples = self.dataset_settings.get("max_samples", None)
        self.streaming = self.dataset_settings.get("streaming", False)

        dataset_source = self._normalize_data_location(self.data_location)
        self.dataset = self._load_dataset(dataset_source)
        self._column_name_map = self._build_column_name_map()
        self._register_getters()
        super().__init__(config)

    def _normalize_data_location(self, data_location: str) -> str:
        if data_location.startswith("hf://"):
            return data_location[5:]
        return data_location

    def _load_dataset(self, dataset_source: str):
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError(
                "MultimodalUniverseDataset requires the `datasets` package. "
                "Install it with `pip install datasets`."
            ) from err

        dataset = load_dataset(dataset_source, split=self.split, streaming=self.streaming)
        dataset = dataset.with_format("numpy")

        if self.streaming:
            if self.max_samples is None:
                raise ValueError(
                    "When streaming=True, set data_set.MultimodalUniverseDataset.max_samples "
                    "to avoid iterating through the full dataset."
                )
            dataset = list(itertools.islice(dataset, int(self.max_samples)))
        elif self.max_samples is not None:
            dataset = self._limit_non_streaming_dataset(dataset, int(self.max_samples))

        return dataset

    def _limit_non_streaming_dataset(self, dataset: Any, max_samples: int):
        limit = min(max_samples, len(dataset))
        if hasattr(dataset, "select"):
            return dataset.select(range(limit))
        if isinstance(dataset, list):
            return dataset[:limit]
        return _IndexedSubset(dataset, limit)

    def _build_column_name_map(self) -> dict[str, str]:
        sample = self.dataset[0]
        column_name_map: dict[str, str] = {}
        for key in sample:
            # Always register the raw key so users can request exact MMU field
            # names (including punctuation) in data_request.fields.
            column_name_map[key] = key

            # Register a sanitized alias for convenience.
            sanitized = self._sanitize_name(key)
            column_name_map.setdefault(sanitized, key)

        return column_name_map

    def _sanitize_name(self, column_name: str) -> str:
        sanitized = re.sub(r"\W", "_", column_name)
        if not sanitized:
            return "field"
        if sanitized[0].isdigit():
            return f"field_{sanitized}"
        return sanitized

    def _register_getters(self) -> None:
        def _make_getter(source_name):
            def getter(self, idx, _source_name=source_name):
                from PIL.Image import Image
                import numpy as np

                retval = self.dataset[idx][_source_name]
                
                # Some fields in MMU are PIL images. 
                # Hyrax only acepts numpy arrays
                if isinstance(retval, Image):
                    retval = np.asarray(retval)
                
                return retval

            return getter

        for method_suffix, source_name in self._column_name_map.items():
            method_name = f"get_{method_suffix}"
            if not hasattr(self, method_name):
                setattr(self, method_name, MethodType(_make_getter(source_name), self))

    def __len__(self) -> int:
        return len(self.dataset)

    # def __getitem__(self, idx: int) -> dict[str, Any]:
    #     return self.dataset[idx]

    # def sample_data(self) -> dict[str, dict[str, Any]]:
    #     return {"data": self.dataset[0]}
