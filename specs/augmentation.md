# Augmentation in Hyrax

Link to sampling doc: [Splits and Dataset Balancing in Hyrax](https://docs.google.com/document/d/11_Ma-hF5u7xV-MBWXvt68ibn-UqCesS9jrXeD8AzOlM/edit?tab=t.0)

## Motivation and Scope:

Most astronomical datasets are unbalanced in the sense that they have many more members of particular classes than others. This presents a challenge for training ML models, which is typically remedied with some combination of focal loss, and sampling or augmenting the data set. Focal loss is already easily incorporated into hyrax via a model's \`train\_batch\` function; however, support within hyrax for balanced sampling and augmentation techniques has been nonexistent. This gap has leadto hyrax users rolling their own independent versions of the functionality ([kbmod-ml](https://github.com/dirac-institute/kbmod-ml/compare/main...htohfa:kbmod-ml:main#diff-8882292c2b762bb56ce29b385a9f01f2a9a26ca808aef439266b250a0f67dd08), [applecider](https://github.com/applecider-ml/applecider/blob/1921a54db8444d4b74d2947c9a7dc68b1790a1ef/src/applecider/datasets/oversampler_mixin.py)).

Like any problem having to do with data preparation, the ultimate fall-back is for a user to prepare a dataset which is already sampled and/or augmented according to their desire, and then to write a dataset class for hyrax which feeds their processed data to their model. This creates significant user experience gaps enumerated below:

**Unfamiliar with ML Techniques:** An astronomer unfamiliar with ML techniques, but familiar enough with ML and their data to identify a class imbalance wants to address a data class imbalance in their Hyrax project. Because Hyrax does not offer an avenue for them, they must learn and implement all of their data sampling and augmentation outside of hyrax. Then they must integrate this data with hyrax, potentially requiring them to learn Hyrax's dataset classes, and data handling in greater detail than would have been otherwise necessary had their dataset not been unbalanced.

**Using a public or built-in dataset:** An astronomer using a public dataset (such as LSSTDownloadedDataset, or MultiModalUniverseDataset) encounters an ML problem where they need to rectify a class imbalance in public data. In the current approach they must either reimplement significant parts of the corresponding hyrax dataset class, or make their augmenting code mimic the public dataset in some way. In most cases they will incur significant storage costs for augmented data at Rubin scale, which would be otherwise unnecessary.

**Sharing augmentation techniques:** In a public dataset where there are unbalanced data and several known successful augmentation techniques, the current paradigm offers no way for dataset authors making that public dataset available to also make the dominant augmentation techniques available to other scientists. Ultimately users of the dataset are forced to implement their own augmentation.

By providing an accessible, performant, and intuitive configuration interface to balanced dataset sampling and augmentation, we can offer all of these users value from using Hyrax. This document will cover the augmentation design only, while balanced sampling concerns will be in [Splits and Dataset Balancing in Hyrax](https://docs.google.com/document/d/11_Ma-hF5u7xV-MBWXvt68ibn-UqCesS9jrXeD8AzOlM/edit?tab=t.0)

There are many methods for data augmentation, some of which are too complex or coupled to specific types of data to support initially. By supporting too many paradigms, we risk bloating the feature and making it too complex. As a first pass we plan to support only augmentation methods where each new sample is created from a single data sample in the underlying dataset.

## Design Overview

We plan to programmatically extend the Hyrax dataset interface to support an `augment_<field_name>` family of functions in parallel to the existing `get_<field_name>`  
family of functions. This allows dataset writers to define default augmentations, which can be used simply by configuration as well as for users of dataset classes to easily override the default augmentation method.

We will also be adding an `augment` configuration option on the friendly name of a Dataset. In the first version this will be a simple `true`/`false` flag to control whether the `augment_<field_name>` functions are used during sampling. In a future version it could be extended to a dictionary that would allow per-field selection of `augment` or `get` methods, allowing easier selection of augmentation methods on pre-built dataset classes.

## Configuration

A data request in hyrax configuration is formed of nested dictionaries, where the outer layer is a data group (e.g. "train", "validate", "test", "infer") and within that there are friendly names which each correspond to a single hyrax dataset class at runtime. 

An example configuration with the new V1 key is below:

`"train": { # data group`  
  `"data": { # friendly name`  
    `"dataset_class": "HyraxCifarDataset",`  
    `"fields": ["image", "label"],`  
    `"primary_id_field": "object_id",`  
    `"augment": true`  
  `}`  
`}`

**`augment`:**   
If true, oversampling/augmentation is enabled, If this section is missing, no augmentation or oversampling is performed, and Hyrax reverts to its exact behavior prior to this feature being implemented. The default value is `false`.

When augmentation is enabled, all fields with `augment_<field_name>` member functions defined will be used, and when the `augment_<field_name>` function is not defined, data access will fall back to the `get_<field_name>`

In Version 2 the `augment` config key can optionally be a list of field names to augment, parallel to the `fields` list in the data request.

`"train": { # data group`  
  `"data": { # friendly name`  
    `"dataset_class": "HyraxCifarDataset",`  
    `"fields": ["image", "label"],`  
    `"primary_id_field": "object_id",`  
    `"augment": ["image"]`  
  `}`  
`}`

Fields listed in `augment` will use `augment_<field_name>` methods. Fields not listed will use `get_<field_name>`. The lack of the corresponding `augment_<field_name>` method for a listed field is a hard error, rather than the permissive system in V1.

The `primary_id_field` must not appear in the `augment` list — it is implicitly repeated for oversampling and must not be augmented. Listing the `primary_id_field` in `augment` is a fatal error.

Fields listed in `augment` must be a subset of `fields`. Listing a field for augmentation that is not in the data request is a fatal error.

If a field is listed in `augment` and the corresponding `augment_<field_name>` method does not exist on the underlying dataset class, the configuration is invalid and Hyrax must stop and produce an informative error. The detection of this class of error ought to be delayed after config parsing and validation (potentially as late as the first valid call of the `augment_<field_name>` method by `DataProvider`) in order to permit metaprogramming by Dataset authors in their constructors.

## Dataset Augmentation Interface

The data augmentation functions on a Hyrax dataset class are modeled after the existing Hyrax `get_<field_name>` methods, in that the name contains the field name to which the function refers. The canonical signature is:

**`def augment_<field_name>(self, data, index, rng_seed):`**

**`self`** is the typical class object reference to the Dataset class.

**`data`** is the result from calling the corresponding `get_<field_name>` method during sampling.  
Because the return value from the `get_<field_name>` functions are cached, but the result of `augment_<field_name>` is not, this interface choice decouples the `rng_seed` generation from the data cache subsystem.

**`index`** is the index of the data in the dataset. This allows authors to implement various index-aware augmentation strategies. See examples below:

**`rng_seed`** is a 64 bit integer which is provided to allow correlation between fields in the same row of augmented data. Hyrax will derive `rng_seed` values from the master random seed (`data_set.seed` in hyrax config) via a two-level numpy RNG chain: a top-level `_augment_rng` advances once per epoch to produce a fresh `_epoch_rng`, which is then drawn from sequentially in `resolve_data` — one integer per call. This means:

- **Single-threaded access** is reproducible by call order within an epoch.
- **Multi-threaded access** is intentionally non-reproducible (no locks on the hot path).
- The same `rng_seed` will be passed to all `augment_<field>` calls within a single row, enabling correlated augmentation across fields (e.g. same rotation for image and mask).

The return value from `augment_<field_name>` is the augmented data.

It is important to note that when augmentation is enabled on a field, **every call to that field (whether it is for a class that is oversampled or not)** will use the `augment_<field_name>` codepath. This is to ensure that all data goes through augmentation, not just the data in classes which are underrepresented in the underlying dataset.

Datasets will also get a new callback `on_epoch_start(self, verb: str)`, which will be implemented by a `pass` implementation in `HyraxDataset` which classes can override. The `verb` parameter will receive the name of the running verb (e.g. `"train"`, `"infer"`, `"test"`, `"engine"`). DataProvider will get a corresponding `on_epoch_start(self, verb: str)` which will reset the epoch RNG and dispatch calls to all active Dataset instances.

For the `train` verb, DataProvider's `on_epoch_start` will be called via an `Events.EPOCH_STARTED` handler registered in `Train.run()`. For single-pass verbs (`infer`, `test`, `engine`), it will be called once before execution begins.

## Dataset Caching interface:

DataCache maintains two separate cache maps per dataset:

* **Base cache** — keyed directly by `real_idx` (an int). Stores the result of `get_<field>` calls. No dataset method is called to produce the key; the index is used as-is.
* **Augment cache** — keyed by the return value of `augment_cache_key`. Stores augmented results. Only populated when the dataset opts in.

By default augmented data is not cached, which is the standard expectation in ML training where augmentations should produce different results each epoch. Dataset authors who want to cache augmented results (e.g., when augmentation is deterministic and expensive) can override:

`def augment_cache_key(self, idx, rng_seed):`  
    `return None`

This method is only called when augmentation is active. It receives the dataset-local index and the `rng_seed` that was passed to `augment_<field>`. It should return an `np.int64` cache key, or `None` to skip caching augmented data.

On lookup, DataCache checks the augment cache first (when augmentation is active and an `rng_seed` is present), then falls back to the base cache. This two-level lookup means a base cache hit still avoids calling `get_<field>` even when the augmented result isn't cached — only `augment_<field>` re-runs.

## Example: Random Rotations

Given an image dataset with a mask layer, object ids and labels the config would look something like the following.

`"train": {`  
  `"data": {`  
    `"dataset_class": "ExampleAugmentedDataset",`  
    `"fields": ["image", "mask", "label"],`  
    `"primary_id_field": "object_id",`  
    `"augment": ["image", "mask"]`  
    `# or "augment": true in v1`  
  `}`  
`}`

This configuration protects `label` and `object_id` fields from augmentation, but `mask` and `image` are subject to random rotations defined by the member functions in `ExampleAugmentingDataset` shown below:

**`import torch`**  
**`import torchvision.transforms.functional as F`**

**`class ExampleAugmentedDataset(ExampleDataset):`**

  **`def random_rotation(self, rng_seed):`**  
    **`gen = torch.Generator().manual_seed(rng_seed)`**  
    **`return torch.empty(1).uniform_(-180,180, generator=gen)[0]`**

  **`def augment_image(self, data, idx, rng_seed):`**  
    **`return F.rotate(data, angle = self.random_rotation(rng_seed))`**

  **`def augment_mask(self, data, idx, rng_seed):`**  
    **`return F.rotate(data, angle = self.random_rotation(rng_seed))`**

In this example, the augment functions receive the same `rng_seed` across each oversampled index from hyrax. This property combined with the management of the rng in the `random_rotation` function ensures that both image and mask receive the same random rotation for a given row of data fed to the model.

When this class is sampled under a balanced regime (e.g. because  [Splits and Dataset Balancing in Hyrax](https://docs.google.com/document/d/11_Ma-hF5u7xV-MBWXvt68ibn-UqCesS9jrXeD8AzOlM/edit?tab=t.0) is enabled) the `augment_image` and `augment_mask` functions receive more than one call with the same `idx` value. This behaviour ensures that a downstream ML model learns to distinguish between the classes, and not distinguish whether the augmentation function was run, because the augmentation function is always run.

Note also that the `ExampleAugmentDataset` class derives from `ExampleDataset` demonstrating how someone might extend a widely distributed dataset class to perform a custom augmentation on a small number of fields relevant to their science.

## Example: Augmenting only repeated data

For some data it is desirable to use the underlying data unchanged on first access and augmentation on subsequent access in an epoch. The random rotation dataset class can be extended as follows to achieve this result using a `self.seen` set that is reset in the `on_epoch_start` method.

**`class ExampleAugmentedDataset(ExampleDataset):`**  
  **`def on_epoch_start(self, verb):`**  
    **`self.seen = set()`**  
    
  **`def apply_rotation(self, data, idx, rng_seed):`**  
    **`if idx in self.seen:`**  
      **`return F.rotate(data, angle=self.random_rotation(rng_seed))`**  
    **`self.seen.add(idx)`**  
    **`return data`**

  **`def random_rotation(self, rng_seed):`**  
    **`gen = torch.Generator().manual_seed(rng_seed)`**  
    **`return torch.empty(1).uniform_(-180,180, generator=gen)[0]`**  
  **`def augment_image(self, data, idx, rng_seed):`**  
    **`return self.apply_rotation(data, idx, rng_seed)`**  
  **`def augment_mask(self, data, idx, rng_seed):`**  
    **`return self.apply_rotation(data, idx, rng_seed`**

## Non-Training Actions

For "infer" data requests defining any augmentation is a hard error (enforced by a model validator on `DataRequestDefinition`), because augmentation/oversampling should never occur in inference on real data.

For "validate" and "test" there are valid reasons for wanting to examine performance on purely augmented data (e.g. Test Time Augmentation), so an `augment` configuration is valid on those groups.

## V1 Implementation Details

The following details reflect design decisions for the V1 implementation:

**Memory safety without deepcopy:** Rather than `deepcopy` of cached base data, the augmentation pass will build a new output dict. Non-augmented fields will share references to the cached arrays. For augmented ndarray fields, a read-only view (`value.view()` with `writeable=False`) will be passed to `augment_<field>`, ensuring augmentation functions cannot mutate the cache. Augmentation functions are expected to return new arrays.

**Tensorboard metrics:** Augmentation time will be logged as `augmentation_s` to tensorboard alongside the existing `cache_hit_s` and `cache_miss_s` metrics.

**Join-map handling:** When augmentation is active on a secondary (joined) dataset, the dataset-local index from the join map will be passed to `augment_<field>` rather than the DataProvider-level index.

**Augment caching deferred to V3:** V1 does not cache augmented results — the existing DataCache caches `get_<field>` results as before, and augmentation is applied post-cache.

## V3 Cache Restructuring

Implementation plan: `specs/augmentation-cache-plan.md`

The following decisions were made for the `augment_cache_key` implementation and associated DataCache restructuring:

**Two separate cache maps per dataset:** DataCache maintains a base cache (`dict[int, dict]`) keyed directly by `real_idx`, and an augment cache (`dict[np.int64, dict]`) keyed by `augment_cache_key`. The separate key-spaces make collisions impossible and eliminate method calls on the base-data hot path — the index is used directly as the cache key with no dispatch.

**`augment_cache_key` opt-in:** The default `augment_cache_key` returns `None` (don't cache augmented data). This matches the ML convention that augmented data should vary across epochs. Dataset authors override this only when augmented results are deterministic and expensive to recompute. The method is only called when augmentation is active, so datasets without augmentation pay zero cost.

**Per-dataset cache maps:** DataCache maintains separate base and augment caches per dataset (friendly name). This replaces the single flat map keyed by DataProvider index, supporting different index mappings (joins) and per-dataset augmented-data caching decisions.

**Preload thread removed:** The `preload_cache` and `preload_threads` config keys and the associated `DataCache` preload thread are removed. The preload thread's strategy of sequentially iterating indices 0 through len does not match the access pattern of `WeightedRandomSampler` (used by the balanced sampling feature). Users needing I/O prefetching should use PyTorch DataLoader's `num_workers` and `prefetch_factor` parameters, which are already passed through from Hyrax's `[data_loader]` config and which naturally match whatever access pattern the sampler dictates.

**Augmentation folded into per-dataset loop:** Instead of a separate augmentation sweep over the assembled data dict (the V1 approach), augmentation is applied per-dataset inside the cache lookup loop in `resolve_data`. This eliminates the second pass and allows cache hits on augmented data (when `augment_cache_key` opts in) to skip both the `get_<field>` and `augment_<field>` calls entirely.