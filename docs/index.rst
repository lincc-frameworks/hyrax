Welcome to Hyrax's documentation!
=================================

Hyrax - A framework for machine learning in astronomy
-----------------------------------------------------
Hyrax is a powerful and extensible machine learning framework that automates data
acquisition, scales seamlessly from laptops to HPC, and ensures reproducibility 
— freeing astronomers to focus on discovery instead of infrastructure.


Getting Started
==================
To get started, we recommend creating a virtual environment (e.g., venv or conda)
and then install Hyrax using ``pip``.

.. code-block:: console

   >> conda create env -n hyrax python=3.12
   >> conda activate hyrax
   >> pip install hyrax

With Hyrax installed in your environment, you can train a built in autoencoder right away.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           from hyrax import Hyrax

           # Create an instance of the Hyrax object
           h = Hyrax()

           # Train the default autoencoder model
           h.train()

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax train

Without any additional specification, the model is trained using the
`CiFAR dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`_.

Once the model is trained, we can immediately use it for inference.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           # Infer on the CiFAR test dataset
           results = h.infer()

           # Access inference results directly in your notebook.
           results[0]

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax infer

Hyrax will automatically use the most recently trained model to infer on the CiFAR test dataset.

The results of inference will be 64 element floating point vectors saved as .npy
files in a timestamped directory under the default ``./results/`` directory, and available in your 
notebook via the returned object.

To visualize the results, we should created an embedded lower dimensional space.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           # Create a 2D UMAP representation of the 64 element vectors
           results = h.umap()

           # Access umapped data points directly in your notebook.
           results[0]

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax umap

The `UMAP <https://umap-learn.readthedocs.io/en/latest/>`_ algorithm is used to
create a 2D representation of the 64 element vectors. These are saved in the same tree as the inference 
results, and are similarly available in your notebook via the returned object.

The results of using UMAP can then be examined with an interactive visualization
withing a Jupyter notebook.

.. code-block:: python

   # Visualize the UMAP results
   h.visualize()

.. figure:: _static/umap_visualization.JPG
   :align: center
   :alt: An example UMAP visualization of the CiFAR dataset.

   An example UMAP visualization of the CiFAR dataset in a Jupyter notebook.

This example shows how Hyrax can be used to quickly train, infer, and visualize
data.
To dive deeper, start with the :doc:`Configuration <configuration>`
section to learn how to configure Hyrax for your specific use case.


.. toctree::
   :hidden:

   Getting started <self>
   About <about>
   Configuration <configuration>
   Verbs <verbs>
   Builtin Datasets <autoapi/hyrax/data_sets/index>
   Custom Datasets and Models <external_libraries>
   Data set splits <data_set_splits>
   Model comparison <model_comparison>
   Architecture overview <architecture_overview>
   Developer guide <dev_guide>
   API Reference <autoapi/index>
   Example notebooks <notebooks>
