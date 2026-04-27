---
name: hyrax-dataset-class
description: Create or update Hyrax dataset classes. Use when the user asks to "create a dataset class", "add a HyraxDataset", "load my data through Hyrax", "add dataset getter methods", "wire dataset defaults", "add dataset tests", or "make a notebook example" for a Hyrax dataset. Also use when a task involves data_request fields, primary_id_field, hyrax_default_config.toml defaults, or dataset code under src/hyrax/datasets.
metadata:
  version: "0.1.0"
---

# Hyrax Dataset Class

## First Read

Read `docs/dataset_class_reference.rst` before implementing. Treat it as the interface contract.

Read `src/hyrax/hyrax_default_config.toml` before adding config keys. Match the existing TOML style and put dataset-specific defaults under `[data_set.<ClassName>]`.

Use existing dataset modules, tests, and notebook examples only as local patterns. Do not blindly copy an implementation; adapt the shape to the user's data format and requested fields.

Read [references/config-rules.md](references/config-rules.md) before reading or adding dataset config keys.

Read [references/test-checklist.md](references/test-checklist.md) before writing or reviewing dataset tests.

## Canonical Minimal Example

Use this as the starting shape for any new dataset class. Adapt it to the user's data format.

```python
from hyrax.datasets import HyraxDataset


class ExampleTabularDataset(HyraxDataset):
    """Hyrax dataset wrapping a tabular data source."""

    def __init__(self, config: dict, data_location=None):
        if data_location is None:
            raise ValueError("A `data_location` must be provided.")

        self.data_location = str(data_location)
        settings = config["data_set"]["ExampleTabularDataset"]
        self.read_kwargs = settings["read_kwargs"]

        self.table = self._load_table()
        super().__init__(config)

    def _load_table(self):
        import some_library
        return some_library.open(self.data_location, **self.read_kwargs)

    def get_object_id(self, idx):
        return str(self.table[idx]["object_id"])

    def get_flux(self, idx):
        return float(self.table[idx]["flux"])

    def __len__(self):
        return len(self.table)
```

Key points this example demonstrates:
- Import from `hyrax.datasets`, not `hyrax.datasets.dataset_registry`.
- Class name ends in `Dataset`.
- One-line class docstring.
- Config access with `[]`, never `.get()`.
- Pass-through kwargs via a config sub-table.
- `super().__init__(config)` called last.
- Optional dependency imported inside the method that uses it.
- Explicit getter methods for known fields.

## User Discovery

Ask only material questions. Get enough information to define the golden path:

- Data location and storage format: file, directory, database, remote service, table name, split name, or other locator.
- Object granularity: what one `idx` represents.
- Dataset length source: catalog rows, files, table rows, remote records, or configured limit.
- Requested Hyrax fields: `fields` values and the `primary_id_field` the data request will use.
- Field shapes and types: scalar, array, image, time series, nested table, label, mask, metadata.
- Required library pass-through kwargs and which options need Hyrax defaults.

If the user is still exploring, stay at their level of generality: sketch the class skeleton, identify missing data details, and implement only what can be grounded in their example or specification.

## Implementation Workflow

1. Choose a class name ending in `Dataset` and a module under `src/hyrax/datasets/`.
2. Import `HyraxDataset` from `hyrax.datasets` (the canonical import path).
3. Implement `__init__(self, config: dict, data_location=None)`.
4. Store `data_location` when relevant and do one-time setup there: locate files, load small catalogs, open handles, or store pass-through kwargs.
5. Call `super().__init__(config)` after dataset-specific setup unless the surrounding local pattern requires otherwise.
6. Implement `__len__(self)`.
7. Implement `get_<field_name>(self, idx)` for every field Hyrax may request, including `get_<primary_id_field>`.
8. Return stable, unique IDs from the primary ID getter. Prefer an existing unique object ID; otherwise use a stable index or deterministic hash from identifying values.
9. Add a one-line class docstring describing what the dataset wraps.
10. Add focused tests under `tests/hyrax/` that create minimal sample data and assert length, primary IDs, requested fields, and config/pass-through behavior.
11. Add a notebook or docs example when the request asks for a user-facing workflow or the dataset format needs demonstration.

Keep heavy per-object work inside getters. Keep constructor work limited to setup that should happen once.

Do not implement `__getitem__` or `collate` unless the user specifically asks for custom batching behavior. The base class and Hyrax machinery handle these.

## Code Style

Optimize for readability of the normal path over exhaustive error correction. Validate only the inputs that would otherwise produce confusing failures or silent wrong results.

Default to explicit getters. Use dynamic getter registration (the `_register_getters` pattern) only when the columns come from the data source at runtime and are not known at development time (e.g. CSV columns, database schemas, Parquet fields). For most scientist-written datasets where the fields are known up front, explicit `get_<field>` methods are clearer and easier to debug.

Keep optional dependency imports close to where they are needed when the dependency is dataset-specific.

Preserve underlying library return types when they are already useful to Hyrax. Convert only when Hyrax or tests need a stable representation, such as `float`, `int`, `str`, NumPy arrays, or tensors.
