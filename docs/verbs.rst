Hyrax Verbs
===========
The term "verb" describes the actions that Hyrax can perform.
Each verb is available as a method on the ``Hyrax`` object in a notebook
and (unless noted) as a subcommand of the ``hyrax`` CLI.


``train``
---------
Train a model. The specific model and training data are specified via the
:doc:`configuration </configuration>` or by calling ``h.set_config()`` after
creating a ``Hyrax`` instance.

Returns the trained ``torch.nn.Module`` in a notebook context.

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

           from hyrax import Hyrax

           h = Hyrax()
           model = h.train()

    .. tab-item:: CLI

        .. code-block:: bash

           >> hyrax train


``infer``
---------
Run inference using a trained model. If no model weights are specified,
Hyrax automatically finds the most recently trained model in the results
directory. You can also choose which :ref:`dataset split <dataset_splits>`
to run inference on.

Returns a ``ResultDataset`` in a notebook context.

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

           h.infer()

    .. tab-item:: CLI

        .. code-block:: bash

           >> hyrax infer


``test``
--------
Evaluate a trained model on test data, computing metrics and logging results
to MLflow.

Returns a ``ResultDataset`` in a notebook context.

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

           h.test()

    .. tab-item:: CLI

        .. code-block:: bash

           >> hyrax test


``umap``
--------
Run `UMAP <https://umap-learn.readthedocs.io>`_ on the output of inference
to reduce high-dimensional embeddings to 2D (or 3D) for visualization. By
default, Hyrax uses the most recent inference output. See the
:doc:`UMAP notebook </pre_executed/using_umap>` for configuration options.

Returns a ``ResultDataset`` containing the reduced embeddings.

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

           h.umap()

    .. tab-item:: CLI

        .. code-block:: bash

           >> hyrax umap [-i <path_to_inference_output>]


``visualize``
-------------
Interactively visualize the embedded space produced by the ``umap`` verb.
Renders an interactive Holoviews/Bokeh scatter plot with linked data table
and optional image thumbnails.

.. note::
   Notebook-only. Not available from the CLI.

.. code-block:: python

    h.visualize(width=800, height=800)


``prepare``
-----------
Load and return the configured datasets without running any training or
inference. Useful for inspecting data, verifying the pipeline, and
prototyping ``prepare_inputs`` or ``collate`` functions.

Returns a dictionary of ``DataProvider`` objects keyed by split name
(e.g. ``"train"``, ``"infer"``).

.. note::
   Notebook-only. Not available from the CLI.

.. code-block:: python

    datasets = h.prepare()
    sample = datasets["train"][0]


``model``
---------
Resolve and return the model *class* (not an instantiated model) from the
current configuration. Useful for inspecting or overriding ``prepare_inputs``
before training.

.. note::
   Notebook-only. Not available from the CLI.

.. code-block:: python

    ModelClass = h.model()


``download``
------------
Download survey cutouts (e.g. from HSC) based on a catalog and configuration.

.. note::
   Notebook-only. Not available from the CLI.

.. code-block:: python

    h.download()


``rebuild_manifest``
--------------------
Rebuild a download manifest file by re-scanning dataset files on disk.
Currently only applicable to ``HSCDataset``.

.. note::
   Notebook-only. Not available from the CLI.

.. code-block:: python

    h.rebuild_manifest()


``lookup``
----------
Look up the inference result for a single object by its ID.

Returns a ``numpy.ndarray`` (the latent vector) or ``None`` if not found.

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

           result = h.lookup(id="object_42")

    .. tab-item:: CLI

        .. code-block:: bash

           >> hyrax lookup -i <object_id> [-r <results_dir>]


``save_to_database``
--------------------
Insert inference results into a vector database for similarity search.
Supports ChromaDB and Qdrant backends (configured via ``[vector_db]``).
By default uses the most recent inference output. See the
:doc:`vector database notebook </pre_executed/vector_db_demo>` for an
end-to-end walkthrough.

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

           h.save_to_database()

    .. tab-item:: CLI

        .. code-block:: bash

           >> hyrax save_to_database [-i <inference_dir> -o <database_dir>]


``database_connection``
-----------------------
Open a connection to an existing vector database for interactive similarity
queries (``search_by_id``, ``search_by_vector``, ``get_by_id``).

Returns a vector database connection object.

.. note::
   Notebook-only. Not available from the CLI.

.. code-block:: python

    db = h.database_connection()
    neighbors = db.search_by_id("object_42", k=5)


``to_onnx``
------------
Export a trained PyTorch model to ONNX format for portable, framework-free
inference via the ``engine`` verb.

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

           h.to_onnx()

    .. tab-item:: CLI

        .. code-block:: bash

           >> hyrax to_onnx [--input-model-directory <dir>]


``engine``
----------
Run inference using an exported ONNX model. Intended for production
deployments that do not require PyTorch at runtime.

.. tab-set::

    .. tab-item:: Notebook

        .. code-block:: python

           h.engine()

    .. tab-item:: CLI

        .. code-block:: bash

           >> hyrax engine [--model-directory <dir>]
