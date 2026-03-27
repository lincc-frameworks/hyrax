External package setup
======================

This page shows the minimum steps to turn notebook classes into a local Python
package that Hyrax can import.

If you want a working example package, see
`external_hyrax_example <https://github.com/lincc-frameworks/external_hyrax_example>`_.


Goal
----

Start from classes that already work in a notebook (the :doc:`dataset class reference </dataset_class_reference>` and :doc:`model class reference </model_class_reference>` cover the required interfaces), then:

1. put them into a package directory,
2. install that package locally,
3. point Hyrax config to fully-qualified class paths.


Step 1: make a minimal package folder
-------------------------------------

From a terminal, create this structure:

.. code-block:: text

   my_hyrax_library/
   ├── pyproject.toml
   └── src/
       └── my_hyrax_library/
           ├── __init__.py
           ├── datasets/
           │   ├── __init__.py
           │   └── my_dataset.py
           └── models/
               ├── __init__.py
               └── my_model.py


Step 2: put a minimal ``pyproject.toml`` in place
-------------------------------------------------

Create ``my_hyrax_library/pyproject.toml``:

.. code-block:: toml

   [build-system]
   requires = ["setuptools>=61", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "my-hyrax-library"
   version = "0.1.0"
   description = "My Hyrax dataset and model classes"
   dependencies = ["hyrax"]

   [tool.setuptools]
   package-dir = {"" = "src"}

   [tool.setuptools.packages.find]
   where = ["src"]


Step 3: move notebook classes into package files
------------------------------------------------

* Copy your dataset class into ``src/my_hyrax_library/datasets/my_dataset.py``.
* Copy your model class into ``src/my_hyrax_library/models/my_model.py``.

In each folder, keep ``__init__.py`` files so Python treats them as packages.


Step 4: install locally for development
---------------------------------------

From inside ``my_hyrax_library/``:

.. code-block:: bash

   pip install -e .

Now Python can import your classes from anywhere in that environment.

Quick check:

.. code-block:: bash

   python -c "from my_hyrax_library.datasets.my_dataset import MyDataset; print(MyDataset)"


Step 5: use full class paths in Hyrax config
---------------------------------------------

Hyrax must be able to import your classes by full path.

Required config keys:

* ``model.name``
* ``data_request.<group>.<friendly_name>.dataset_class``
* ``primary_id_field`` for each dataset definition

Copy-paste example:

.. code-block:: python

   data_request = {
       "train": {
           "science": {
               "dataset_class": "my_hyrax_library.datasets.my_dataset.MyDataset",
               "data_location": "/path/to/data",
               "fields": ["flux", "label", "object_id"],
               "primary_id_field": "object_id",
           }
       },
       "infer": {
           "science": {
               "dataset_class": "my_hyrax_library.datasets.my_dataset.MyDataset",
               "data_location": "/path/to/data",
               "fields": ["flux", "object_id"],
               "primary_id_field": "object_id",
           }
       },
   }

   h.set_config("data_request", data_request)
   h.set_config("model.name", "my_hyrax_library.models.my_model.MyModel")


Step 6: keep package-specific config values in ``default_config.toml``
-------------------------------------------------------------------------

A beginner-friendly pattern is to ship defaults in your package and let users
change only what they need. Hyrax merges these with its own defaults using the
layered :doc:`configuration system </configuration_system>`.

Create ``src/my_hyrax_library/default_config.toml``:

.. code-block:: toml

   [my_hyrax_library.MyModel]
   hidden_size = 64
   num_classes = 10

Then read those values in ``MyModel.__init__``:

.. code-block:: python

   hidden = config["my_hyrax_library"]["MyModel"]["hidden_size"]
   out_dim = config["my_hyrax_library"]["MyModel"]["num_classes"]


What to do after local testing
------------------------------

After this works locally:

1. push the package to GitHub,
2. optionally publish to PyPI,
3. keep using the same full class paths in Hyrax notebooks.

The import paths do not change as long as your package/module/class names stay the same.

Beginner-friendly packaging references
--------------------------------------

* Python Packaging User Guide (official):
  `Packaging Python Projects <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_
* setuptools quickstart (official):
  `setuptools: Quickstart <https://setuptools.pypa.io/en/latest/userguide/quickstart.html>`_
