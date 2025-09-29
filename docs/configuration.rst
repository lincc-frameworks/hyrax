Configuration
=============

Hyrax ships with a complete default configuration file that can be used immediately
to run the software, however, to make the most of Hyrax you'll need to modify
the configuration to suit your specific needs.


Using the configuration system
------------------------------
When creating an instance of a ``Hyrax`` object in a notebook or running ``hyrax``
from the command line, the configuration is the primary method for specifying the parameters.

If no configuration file is specified, :ref:`the default<complete_default_config>`
will be used. To specify a different configuration file, use the
``-c | --runtime-config`` flag from the CLI
or pass the path to the configuration file when creating a ``Hyrax`` object.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           from hyrax import Hyrax

           # Create an instance of the Hyrax object
           f = Hyrax(config_file=<path_to_config_file.toml>)

           # Train the model specified in the configuration file
           f.train()

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax train -c <path_to_config_file.toml>


Your first custom configuration
...............................

You could create a copy of the entire default configuration file and modify it to suit
your needs, however that's typically not required because often there are only
a few parameters that must be updated for any given Hyrax action.

If a specific configuration file is provided, Hyrax will combine it with the default
configuration and overwrite the default values with the specific ones.

For example, if a file called ``my_config.toml`` had the following contents:

.. code-block:: bash
   :linenos:

   [general]
       log_level = "debug"

It could be used to override the default ``log_level`` configuration, while leaving
the rest of the configuration unchanged.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           from hyrax import Hyrax

           # Create an instance of the Hyrax object
           f = Hyrax(config_file=my_config.toml)

           # Train the model specified in the configuration file
           f.train()

    .. group-tab:: CLI

        .. code-block:: bash

           >> hyrax train -c my_config.toml


Updating settings in a notebook
...............................
Additionally, Hyrax supports modification of the configuration interactively in a notebook.

.. code-block:: python

   from hyrax import Hyrax

   # Create a Hyrax instance, implicitly using the default configuration
   f = Hyrax()

   # Set the log level for the Hyrax instance config
   f.config['general']['log_level'] = 'debug'

   # Train the model specified in the configuration file
   f.train()


Immutable configurations
........................
Once Hyrax begins running an action, the configuration becomes immutable.
This means that the configuration cannot be changed during the execution of an action,
and attempting to do so in code will raise an exception.

By making the configuration immutable during execution, we ensure that the state
of all parameters can be accurately saved with the results of the action.


About the default configuration
-------------------------------

The default configuration for Hyrax contains safe default values for all of the
settings that Hyrax uses. A portion of the default configuration file is shown below.

.. note::
   Only the first portion of the default configuration file is shown below.
   The entire file can be found at the bottom of the page here: :ref:`complete_default_config`.

   .. literalinclude:: ../src/hyrax/hyrax_default_config.toml
      :language: text
      :linenos:
      :lines: 1-25

There is a lot of information there, but don't worry, we'll break it down for you.

First, the file formatted using TOML for its easy readability and because it is
one of the few markdown languages that natively support comments.
TOML files are organized into "tables", and each table contains one or more
key/value pairs.

For instance the ``[general]`` table (the first table in the config)
contains several keys including ``log_level`` and ``results_dir``.
Each of those keys has a value associated with it.
e.g. ``log_level = "info"``.

Second, every key has an associated comment describing what the key does.
We attempt to keep the comments as concise as possible.

Finally, the configuration file is organized into tables that roughly correspond
to the different actions that Hyrax can take.
For instance, the ``[train]`` table contains parameters needed when training a
model such as ``epochs`` and ``weights_filename``.
While the ``[infer]`` table contains keys such as ``model_weights_file``.


.. _complete_default_config:

Complete default configuration file
-----------------------------------

.. literalinclude:: ../src/hyrax/hyrax_default_config.toml
   :language: bash
   :linenos:
