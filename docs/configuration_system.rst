The ``Hyrax`` Configuration System
==================================

``hyrax`` makes extensive use of the config variables to manage the runtime environment of training and inference runs. There is a ``hyrax_default_config.toml``  file (full contents listed here), included with ``hyrax``, that contains every variable that ``hyrax`` could need to operate. To create a custom configuration file, simply create a ``.toml`` file and change variables as you see fit, or if you’re running with a custom dataset or model, add your own variables. 

Config variables are inherited from a hierarchy of sources, similar to ``python`` classes. First, ``hyrax`` will prioritize the variables set in the default configuration. Next, it will load the relevant default config of any custom ``hyrax`` packages that the user is utilizing. It determines what packages to include by checking what custom classes are loaded in initially and looking for the relevant default configs. If a package doesn’t have a default, ``hyrax`` will throw a warning. Finally, it will use whatever variables have been declared in the user defined config toml (see here for how to load those through a notebook/script or the CLI).

.. figure:: _static/hyrax_config_system.png
   :width: 100%
   :alt: The inheritance hierarchy of the hyrax configuration system.

``hyrax`` will pass along all the configuration variables to the relevant models and dataset classes and allows them to configure the runtime through one system. This allows for extensibility and cross-compatibility within the broader “hyrax ecosystem”. From the point of view of the code, these configuration variables should be static. This makes it easier for researchers to develop code separate from the runtime environment.

A core design principle of ``hyrax`` is "code by config", meaning that all runtime parameters should be set through configuration files rather than hard-coded values. This approach enhances flexibility, reproducibility, and ease of experimentation, as users can modify configurations without altering the underlying codebase. This also facilitates sharing and collaboration, as configurations can be easily shared and adapted for different use cases while keeping fundamental models and datasets consistent.

Typed configuration schemas
--------------------------

Hyrax is introducing typed configuration models using Pydantic for safer validation and
better documentation. The first of these models formalizes the ``[data_request]`` table
used to describe datasets for training, validation, and inference. Two key schemas are
now available in :mod:`hyrax.config_schemas`:

* ``DataRequestConfig`` – defines per-dataset settings (``dataset_class``, ``data_location``,
  ``fields``, ``primary_id_field``, and ``dataset_config``).
* ``DataRequestDefinition`` – wraps the full ``data_request`` table, supporting
  ``train``, ``validate``, ``infer``, and additional dataset keys.

These models provide validation and type safety for dataset configuration structures.
Backward compatibility for the legacy ``[model_inputs]`` table name is maintained at
the configuration loading layer.

Compact examples
----------------

Below are small, Hyrax-focused snippets showing how the ``[data_request]`` table maps
between TOML and the Pydantic models:

* **TOML config**

  .. code-block:: toml

     [data_request.train]
     dataset_class = "HyraxRandomDataset"
     data_location = "./data/train.parquet"
     primary_id_field = "object_id"
     fields = ["object_id", "flux"]

     [data_request.train.dataset_config]
     seed = 42
     num_rows = 100

* **Python (Pydantic) equivalent**

  .. code-block:: python

     from hyrax.config_schemas.data_request import DataRequestDefinition

     cfg = DataRequestDefinition.model_validate(
         {
             "train": {
                 "dataset_class": "HyraxRandomDataset",
                 "data_location": "./data/train.parquet",
                 "primary_id_field": "object_id",
                 "fields": ["object_id", "flux"],
                 "dataset_config": {"seed": 42, "num_rows": 100},
             }
         }
     )

* **Directly set on a ``Hyrax`` instance**

  .. code-block:: python

     import hyrax
     from hyrax.config_schemas.data_request import DataRequestDefinition

     h = hyrax.Hyrax()
     h.set_config("data_request", cfg)

* **Inline construction + ``set_config``**

  .. code-block:: python

     import hyrax
     from hyrax.config_schemas.data_request import DataRequestConfig, DataRequestDefinition

     h = hyrax.Hyrax()
     h.set_config(
         "data_request",
         DataRequestDefinition(
             train=DataRequestConfig(
                 dataset_class="HyraxRandomDataset",
                 data_location="./data/train.parquet",
                 primary_id_field="object_id",
                 fields=["object_id", "flux"],
                 dataset_config={"seed": 42, "num_rows": 100},
             )
         ),
     )

The ``DataRequestDefinition`` model accepts additional dataset keys (e.g., ``validate``
or ``infer``) in the same shape as ``train``. If you still use the legacy
``[model_inputs]`` table name, Hyrax aliases it internally to ``[data_request]`` so
existing configs continue to load.

After training is completed, ``hyrax`` will write out all of the variables (combined from all the various source configs) used at runtime in the runtime directory as a ``runtime_config.toml`` file, so that the user can see what variables were actually used in one place.
