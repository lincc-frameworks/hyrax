# Config Rules

Access Hyrax-owned config keys with `[]`, not `.get()`.

```python
settings = config["dataset"]["YourDatasetClass"]
value = settings["value_with_default"]
```

Add every Hyrax-owned key the dataset reads to `src/hyrax/hyrax_default_config.toml`. Missing defaults should fail loudly with `KeyError` during development.

Use pass-through dictionaries for optional keyword arguments owned by an underlying library. Add a default empty table for the pass-through dictionary, then forward it with `**kwargs`.

```toml
[dataset.YourDatasetClass]
required_option = false

[dataset.YourDatasetClass.open_kwargs]
# library_option = "example"
```

```python
settings = config["dataset"]["YourDatasetClass"]
open_kwargs = settings["open_kwargs"]
resource = library.open(data_location, **open_kwargs)
```

Avoid defensive branches around config keys that Hyrax owns. Use clear branches only for meaningful user choices, such as a `false` default meaning "disabled" or "infer automatically."

Do not redefine optional keyword argument keys owned by an underlying library in Hyrax config. Keep those inside the pass-through dictionary unless Hyrax itself needs to interpret the option.

## Complete TOML Example

This is a real example from `hyrax_default_config.toml`. Use this as a template for new dataset defaults:

```toml
[dataset.LanceDBDataset]
# Table name to open inside the LanceDB database.
table_name = false

# Extra keyword arguments passed directly to lancedb.connect.
[dataset.LanceDBDataset.connect_kwargs]

# Extra keyword arguments passed directly to db.open_table.
[dataset.LanceDBDataset.open_table_kwargs]
```

Key formatting rules:
- `false` is used as a sentinel meaning "not set" (TOML has no null).
- Empty sub-tables (like `connect_kwargs` above) define pass-through dictionaries that start as `{}`.
- Comments above keys explain the option. Commented-out key examples show valid values.
