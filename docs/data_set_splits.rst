.. _data_set_splits:

Data set splits (subsets)
=============================

Datasets used in machine learning are typically split in order to avoid overfitting a particular dataset of 
interest, and to perform various sorts of checking that the model is learning what the researcher intends. 
In Hyrax there are default conventions for splitting data, which can be configured to the liking of the 
investigator.

Splits in training
------------------
By default input datasets are split into train (60%), test (20%), and validate (20%). The ``train`` verb uses 
the train split to train and validate splits to create a validation loss statistic every training epoch. The 
test split is explicitly left out of training.

The size of these splits can be configured in the ``[data_set]`` section of the configuration using the 
``train_size``, ``validate_size``, and ``test_size`` configuration keys. The value is either a number of data points
or a ratio of the dataset, where 1.0 represents the entire dataset. For example:

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

            from hyrax import Hyrax
            h = Hyrax()
            h.config["data_set"]["train_size"] = 0.6
            h.config["data_set"]["validate_size"] = 0.2
            h.config["data_set"]["test_size"] = 0.2

    .. group-tab:: CLI

        .. code-block:: bash
            
            $ cat hyrax_config.toml

            [data_set]
            train_size = 600
            validate_size = 200
            test_size = 200


It is recommended that all three are provided; however, zeroing some out can create different training effects

* If the size of the validate split is zero, then training won't include a validate step.

* If the size of the test split is zero, then all data will be used in the training process as either training data or for validation.

* If the size of the test split is zero and the validate split is zero, training will be run on the entire dataset.


Splits in inference
-------------------

By default the ``infer`` verb uses the entire dataset for inference; however any of the splits can be used by 
specifying the ``[infer]`` ``split`` config value. Valid values are any of the three splits. For example, to 
infer on only the test split:

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

            from hyrax import Hyrax
            h = Hyrax()
            h.config["infer"]["split"] = "test"

            h.infer()

    .. group-tab:: CLI

        .. code-block:: bash

            $ cat hyrax_config.toml
            [infer]
            split = test

            $ hyrax infer -c hyrax_config.toml


Randomness in splits
--------------------

The membership in each split is determined randomly. By default, system entropy is used to seed the random number generator for this purpose. 

You can specify a random seed with the ``[data_set]`` ``seed`` configuration key as follows:

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

            from hyrax import Hyrax
            h = Hyrax()
            h.config["data_set"]["seed"] = 1

    .. group-tab:: CLI

        .. code-block:: bash

            $ cat hyrax_config.toml
            [data_set]
            seed = 1