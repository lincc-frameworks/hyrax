# Config Rules

Access Hyrax-owned config keys with `[]`, not `.get()`.

```python
settings = config["data_set"]["YourDatasetClass"]
value = settings["value_with_default"]
```

Add every Hyrax-owned key the dataset reads to `src/hyrax/hyrax_default_config.toml`. Missing defaults should fail loudly with `KeyError` during development.

Use pass-through dictionaries for optional keyword arguments owned by an underlying library. Add a default empty table for the pass-through dictionary, then forward it with `**kwargs`.

```toml
[data_set.YourDatasetClass]
required_option = false

[data_set.YourDatasetClass.open_kwargs]
# library_option = "example"
```

```python
settings = config["data_set"]["YourDatasetClass"]
open_kwargs = settings["open_kwargs"]
resource = library.open(data_location, **open_kwargs)
```

Avoid defensive branches around config keys that Hyrax owns. Use clear branches only for meaningful user choices, such as a `false` default meaning "disabled" or "infer automatically."

Do not redefine optional keyword argument keys owned by an underlying library in Hyrax config. Keep those inside the pass-through dictionary unless Hyrax itself needs to interpret the option.
