.. _dataset_splits:

Data set splits (subsets)
=============================

Datasets used in machine learning are typically split in order to avoid overfitting a particular dataset of
interest, and to perform various sorts of checking that the model is learning what the researcher intends.
In Hyrax, splits are defined via ``split_fraction`` in the ``data_request`` configuration.

Splits in training
------------------

To split a dataset between training and validation, define named groups in the ``data_request`` that point
to the **same** ``data_location`` and assign each group a ``split_fraction``. Hyrax will partition the
dataset so that each group receives a non-overlapping subset of the data proportional to its fraction.
The fractions for all groups sharing a ``data_location`` must sum to ``<= 1.0``.

For example, to use 80% of the data for training and 20% for validation:

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

            from hyrax import Hyrax
            h = Hyrax()

            data_request = {
                "train": {
                    "my_data": {
                        "dataset_class": "HyraxCifarDataset",
                        "data_location": "./all_data",
                        "primary_id_field": "object_id",
                        "split_fraction": 0.8,
                    }
                },
                "validate": {
                    "my_data": {
                        "dataset_class": "HyraxCifarDataset",
                        "data_location": "./all_data",
                        "primary_id_field": "object_id",
                        "split_fraction": 0.2,
                    }
                },
            }
            h.set_config("data_request", data_request)

    .. tab-item:: CLI

        .. code-block:: toml

            [data_request.train.my_data]
            dataset_class = "HyraxCifarDataset"
            data_location = "./all_data"
            primary_id_field = "object_id"
            split_fraction = 0.8

            [data_request.validate.my_data]
            dataset_class = "HyraxCifarDataset"
            data_location = "./all_data"
            primary_id_field = "object_id"
            split_fraction = 0.2

If you have separate data files for training and validation, simply omit ``split_fraction`` and
point each group to its own ``data_location``:

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

            data_request = {
                "train": {
                    "my_data": {
                        "dataset_class": "HyraxCifarDataset",
                        "data_location": "./train_data",
                        "primary_id_field": "object_id",
                    }
                },
                "validate": {
                    "my_data": {
                        "dataset_class": "HyraxCifarDataset",
                        "data_location": "./validate_data",
                        "primary_id_field": "object_id",
                    }
                },
            }
            h.set_config("data_request", data_request)

    .. tab-item:: CLI

        .. code-block:: toml

            [data_request.train.my_data]
            dataset_class = "HyraxCifarDataset"
            data_location = "./train_data"
            primary_id_field = "object_id"

            [data_request.validate.my_data]
            dataset_class = "HyraxCifarDataset"
            data_location = "./validate_data"
            primary_id_field = "object_id"

The ``train`` :doc:`verb </verbs>` trains on the ``train`` group and, when present, computes a
validation loss each epoch using the ``validate`` group. Adding a ``test`` group is supported
but the train verb does not use it during training — it is available for downstream evaluation.


Randomness in splits
--------------------

When ``split_fraction`` is used, Hyrax randomly assigns indices to each group. By default,
system entropy seeds the random number generator. For reproducible splits, set the ``seed``
key in ``[data_set]``:

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

            from hyrax import Hyrax
            h = Hyrax()
            h.config["data_set"]["seed"] = 1

    .. tab-item:: CLI

        .. code-block:: bash

            $ cat hyrax_config.toml
            [data_set]
            seed = 1
