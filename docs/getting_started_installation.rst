Installation
============

Hyrax can be installed directly from PyPI with ``pip`` or you can install the
pre-release version from GitHub.


.. _install-in-a-virtual-environment:

Install in a virtual environment
---------------------------------

.. tip::

     Most astronomers use conda to manage environments. If you do not already
     have conda installed, see the Anaconda documentation for installing
     `Anaconda <https://www.anaconda.com/docs/getting-started/anaconda/install>`_
     or `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/install>`_.
     Miniconda is the smaller install.

Create and activate a fresh environment, then install Hyrax with ``pip``:

.. code-block:: console

   >> conda create -n hyrax python=3.12
   >> conda activate hyrax
   >> pip install hyrax

Using a dedicated environment helps avoid dependency conflicts with other
Python packages you may already have installed.

.. note::

   Hyrax is officially supported and tested with Python 3.11, 3.12, and 3.13.
   Other versions may work, but are not guaranteed to be compatible.


Install from source
-------------------

.. warning::
      If you install the pre-release version of Hyrax from GitHub, be prepared
      for sudden breaking changes. Hyrax is still at a pre-1.0 release, so the
      development version may change quickly. For best results, stay in touch
      with the Hyrax developers.

If you want the latest development version of Hyrax, or if you plan to edit the
code locally, install from source:

.. code-block:: console

   >> conda create -n hyrax python=3.12
   >> conda activate hyrax
   >> git clone https://github.com/lincc-frameworks/hyrax.git
   >> cd hyrax
   >> pip install -e .


If you are contributing to Hyrax or building the documentation locally, install
the development dependencies as well:

.. code-block:: console

   >> pip install -e '.[dev]'

On some systems, the quotes around ``'.[dev]'`` may not be necessary.
