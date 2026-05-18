# Train Shuffle Configuration

## Summary

Hyrax's `shuffle` config option is train-only. It controls the sampler used for
the training dataloader, not the split-fraction partitioning performed while
constructing datasets.

## User-facing config

```toml
[train]
# If true, shuffle training samples each epoch. Only the train verb uses this option.
shuffle = true
```

`[data_loader].shuffle` is deprecated and migrated to `[train].shuffle` for
versioned user configs.

## Behavior

- `train.shuffle = true` is the default.
- The train verb is the only verb that reads `train.shuffle`.
- In the train verb, the train dataloader uses `SubsetRandomSampler` when
  `train.shuffle = true`.
- In the train verb, the train dataloader uses `SubsetSequentialSampler` when
  `train.shuffle = false`.
- Validation and test dataloaders created by the train verb remain sequential.
- Non-train verbs ignore `train.shuffle` and keep deterministic order.
- Hyrax controls ordering through samplers; `dist_data_loader` does not pass a
  `shuffle` keyword through to PyTorch's `DataLoader`. If a legacy
  `[data_loader].shuffle` key is present, Hyrax warns that the key is ignored.

## Split-fraction note

`setup_dataset(..., shuffle=...)` remains separate from `train.shuffle`. The
former controls whether `split_fraction` indices are shuffled before slicing
partitions. The latter controls sample order while iterating the training
dataloader.

## Migration

A config migration moves:

```toml
[data_loader]
shuffle = false
```

to:

```toml
[train]
shuffle = false
```

The migration records the rename as `data_loader.shuffle -> train.shuffle` so
runtime `set_config` calls that use the old key can emit a deprecation warning.
