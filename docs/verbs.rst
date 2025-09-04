Hyrax Verbs
===========
The term "verb" is used to describe the functions that Hyrax supports.
For instance, the ``train`` verb is used to train a model.
Each of the builtin verbs are detailed here.


``train``
---------
Train a model. The specific model to train and the data used for training is
specified in the configuration file or by updating the default configurations
after creating an instance of the Hyrax object.

When called from a notebook or python, ``train()`` returns a trained pytorch 
model which you can :doc:`immediately evaluate, inspect, or export<pre_executed/export_model>`. Batch evaluations of datasets 
are enabled using the ``infer`` verb, see below.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           from hyrax import Hyrax

           # Create an instance of the Hyrax object
           h = Hyrax()

           # Train the model specified in the configuration file
           model = h.train()

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax train


``infer``
---------
Run inference using a trained model. The specific model to use for inference can
be specified in the configuration file. If no model is specified, Hyrax will find
the most recently trained model in the results directory and use that for inference.
The data used for inference is also specified in the configuration file.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           from hyrax import Hyrax

           # Create an instance of the Hyrax object
           h = Hyrax()

           # Pass data through a trained model to produce embeddings or predictions.
           h.infer()

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax infer

When running infer in a notebook context, the infer verb returns an 
:doc:`InferenceDataSet<autoapi/hyrax/data_sets/inference_dataset/index>` object which can be accessed using 
the ``[]`` operators in python.

``umap``
--------
Run UMAP on the output of inference or a dataset. By default, Hyrax will use the
most recently generated output from the ``infer`` verb.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           from hyrax import Hyrax

           # Create an instance of the Hyrax object
           h = Hyrax()

           # Train a UMAP and process the entire dataset.
           h.umap()

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax umap


``visualize``
-------------
Interactively visualize embedded space produced by UMAP.
Due to the fact that the visualization is interactive, it is not available in the CLI.

.. code-block:: python

    from hyrax import Hyrax

    # Create an instance of the Hyrax object
    h = Hyrax()

    # Train the model specified in the configuration file
    h.visualize()


``prepare``
-----------
Create and return an instance of a Hyrax dataset object. This allows for convenient
investigation of the dataset. While this can be run from the CLI, it is primarily
intended for use in a notebook environment for exploration and debugging.

.. code-block:: python

    from hyrax import Hyrax

    # Create an instance of the Hyrax object
    h = Hyrax()

    # Prepare the dataset for exploration
    dataset = h.prepare()


``index``
---------
Builds a vector database index from the output of inference. By default, Hyrax
will use the most recently generated output from the ``infer`` verb, and will
write the resulting database to a new timestamped directory under the default
``./results/`` directory with the form <timestamp>-index-<uid>.

An existing database directory can be specified in order to add more vectors to
an existing index.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

            from hyrax import Hyrax

            # Create an instance of the Hyrax object
            h = Hyrax()

            # Build a vector database index from the output of inference
            h.index()

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax index [-i <path_to_inference_output> -o <path_to_database_directory>]