Converting your project to Hyrax
================================

The basic flow for using Hyrax in your machine learning project is to:

1. Write code that connects your data and model to hyrax in a notebook.
2. Put working code into a small Python package.
3. Reuse that package across many Hyrax notebooks.

The pages below walk you through making custom dataset and model classes for your data in your project
as well as how to distribute working datasets and models via a python package which can be used by
your collaborators. If you are deciding what to implement first, see :doc:`required_input` for the
minimum pieces Hyrax needs from users.

.. toctree::
   :maxdepth: 1

   Dataset class reference <dataset_class_reference>
   Dataset class notebook example <pre_executed/external_dataset_class>
   Model class reference <model_class_reference>
   Model class notebook example <pre_executed/external_model_class>
   External package setup <external_library_package>

If you want a concrete example package, see
`external_hyrax_example <https://github.com/lincc-frameworks/external_hyrax_example>`_.
