# Removing legacy splitting configs from hyrax

The splitting configuration in Hyrax is too complicated, and as part of preparation for [Splits and Dataset Balancing in Hyrax](https://docs.google.com/document/d/11_Ma-hF5u7xV-MBWXvt68ibn-UqCesS9jrXeD8AzOlM/edit?tab=t.0#heading=h.fh5l9lmtvnce) we are removing the oldest way of specifying dataset splits.

Before we had config migrations, and very early in hyrax's evolution we created a configuration for specifying the manner of splitting a dataset between train, validate, and test. In the configuration toml this looked like (default values shown):

`[data_set]`  
``# Size of the train split. If `false`, the value is automatically set to the``  
`# complement of test_size plus validate_size (if any).`  
`train_size = 0.6`

``# Size of the validation split. If `false`, and both train_size and test_size``  
`# are defined, the value is set to the complement of the other two sizes summed.`  
``# If `false`, and only one of the other sizes is defined, no validate split is created.``  
`validate_size = 0.2`

``# Size of the test split. If `false`, the value is set to the complement of train_size plus``  
``# the validate_size (if any). If `false` and `train_size = false`, test_size is set to `0.25`.``  
`test_size = 0.2`

The only supported way to do this sort of thing after this change will be to use the `split_fraction` configuration within the named data groups of a data request as in the example below, which uses the configuration interface intended for python notebooks:

**`from hyrax import Hyrax`**  
**`h = Hyrax()`**

**`data_request = {`**  
    **`"train": {`**  
        **`"my_data": {`**  
            **`"dataset_class": "HyraxCifarDataset",`**  
            **`"data_location": "./all_data",`**  
            **`"primary_id_field": "object_id",`**  
            **`"split_fraction": 0.8,  # <- Optionally specify a split fraction to split the data into train/validate`**  
        **`}`**  
    **`},`**  
    **`"validate": {`**  
        **`"my_data": {`**  
            **`"dataset_class": "HyraxCifarDataset",`**  
            **`"data_location": "./all_data",`**  
            **`"primary_id_field": "object_id",`**  
            **`"split_fraction": 0.2,  # <- The split fractions for all active groups sharing a data_location must sum to <= 1.0`**  
        **`}`**  
    **`},`**  
**`}`**  
**`h.set_config("data_request", data_request)`**

Both toml and the notebook method shown are equivalent ways to provide configuration information to hyrax; however this proposal is about removing the `data_set.<stage>_size` configurations entirely so that only `split_fraction` in the `data_request` dictionary remains.

The `data_set.<stage>_size` configuration method has been deprecated for over 6 months with many warnings throughout the code. This proposal defines how these configuration parameters will be handled by hyrax now that they are fully removed.

## Hyrax Changes

If hyrax receives the old `data_set.<stage>size` configurations, it will now be a fatal error that explains how to move this information into the `data_request` dictionary as `split_fraction`'s. All verbs that could use the old configurations will emit this fatal error.

All code other than the fatal error described above that accommodates the existence of the old `data_set.<stage>size` configurations will be removed, and in general code surrounding the removed code should be written as if the assumptions of this proposal were always true.

Additionally all tests, example notebooks, documentation and explanatory comments that reference the old way of doing splits should be updated to reflect the new behavior where `split_fraction` in a `data_request` is the only way to specify a split.

