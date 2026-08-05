Dataset class reference
=======================

This page is the ground truth for writing a dataset class for Hyrax.

If you are an astronomer who is new to class-based code, use this as a
copy-and-edit guide.


How Hyrax uses your dataset class
---------------------------------

Hyrax creates your class like this:

.. code-block:: python

   dataset = YourDataset(config=..., data_location=...)

Then, for each object index, Hyrax calls methods named ``get_*``.

The fields Hyrax asks for come from ``data_request`` (see the
:doc:`data requests notebook </notebooks/data_requests>` for how to define one).
Here is a full minimal example for training:

.. code-block:: python

   data_request = {
       "train": {
           "science": {
               "dataset_class": "my_package.datasets.my_dataset.MyDataset",
               "data_location": "/path/to/data",
               "fields": ["flux", "label", "object_id"],
               "primary_id_field": "object_id",
           }
       }
   }

If ``fields`` is ``["flux", "label", "object_id"]``, Hyrax will call:

* ``get_flux(idx)``
* ``get_label(idx)``
* ``get_object_id(idx)``

For a broader discussion of how dataset outputs move through ``collate`` and
``prepare_inputs`` before reaching the model, see :doc:`data_flow`.


Required methods (checklist)
----------------------------

Your class must have all of these:

1. Inherit from ``hyrax.datasets.HyraxDataset``.
2. ``__init__(self, config, data_location=None)`` with ``super().__init__(config)``.
3. ``__len__(self)``.
4. ``get_<field_name>(self, idx)`` for every field listed in ``fields``.
5. ``get_<primary_id_field>(self, idx)`` matching ``primary_id_field`` in config.


Method-by-method guide
----------------------

``__init__(self, config, data_location=None)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

What to do in this method:

1. Save ``data_location``.
2. Do one-time startup work needed by your getters:

   * locate files or verify paths
   * load catalogs if they are reasonably small
   * open remote connections if your data are remote

3. Keep heavy per-object work out of ``__init__``. Put per-object work in
   ``get_*`` methods.
4. Call ``super().__init__(config)`` at the end.

Example (only this method shown):

.. code-block:: python

   def __init__(self, config, data_location=None):
       self.data_location = data_location
       self.catalog = ...
       # Optional: verify data directory exists here
       super().__init__(config)


``__len__(self)``
^^^^^^^^^^^^^^^^^

Return how many objects are in your dataset.

Example:

.. code-block:: python

   def __len__(self):
       return len(self.catalog)


``get_object_id(self, idx)`` (or your chosen ``primary_id_field``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is very important. Hyrax uses this ID to track outputs.

Requirement: IDs should be unique inside your dataset.

If your data already have a unique ID column:

.. code-block:: python

   def get_object_id(self, idx):
       return str(self.catalog[idx]["source_id"])

If your data do not have a unique ID column, two common choices are:

1. Use a running integer.

.. code-block:: python

   def get_object_id(self, idx):
       return str(idx)

2. Build a stable hash from values that identify the object.

.. code-block:: python

   import hashlib

   def get_object_id(self, idx):
       row = self.catalog[idx]
       text = f"{row['ra']:.8f}_{row['dec']:.8f}_{row['mjd_ref']:.2f}"
       return hashlib.sha1(text.encode("utf-8")).hexdigest()


General getter pattern: ``get_<field_name>(self, idx)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the main pattern for all science data (spectra, light curves, images,
scalar parameters, masks, etc.).

Example for a flux vector field:

.. code-block:: python

   def get_flux(self, idx):
       return self.flux_arrays[idx].astype("float32")

Example for a scalar redshift field:

.. code-block:: python

   def get_redshift(self, idx):
       return float(self.photoz[idx])

If you include ``"flux"`` or ``"redshift"`` in ``fields``, Hyrax will call
these methods automatically.


``get_label(self, idx)`` (only when needed)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this for supervised tasks.

If you are doing self-supervised or unsupervised work, you may not need labels.

Example:

.. code-block:: python

   def get_label(self, idx):
       return int(self.labels[idx])


Optional methods
----------------

``collate(self, samples)``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Write this only when default batching is not enough. See the
:doc:`custom collation notebook </notebooks/custom_dataset_collation>` for a runnable
walkthrough.

A common astronomy case is variable-length light curves. The example below pads
all light curves to the longest one in the batch and returns a mask where:

* ``1`` means real data
* ``0`` means padding

Input format:

* ``samples`` is a list like ``[{"data": {...}}, {"data": {...}}]``

Required output format:

* return a dictionary with top-level key ``"data"``

Example:

.. code-block:: python

   import numpy as np

   def collate(self, samples):
       curves = [s["data"]["light_curve"] for s in samples]
       max_len = max(len(c) for c in curves)

       padded = np.zeros((len(curves), max_len), dtype=np.float32)
       mask = np.zeros((len(curves), max_len), dtype=np.float32)

       for i, curve in enumerate(curves):
           n = len(curve)
           padded[i, :n] = curve
           mask[i, :n] = 1.0

       return {"data": {"light_curve": padded, "light_curve_mask": mask}}

When multiple fields need to be collated, it may be preferred to give each field
its own collate function. The :doc:`custom collation notebook </notebooks/custom_dataset_collation>`
contains information on both dataset-level collation and field-level collation.


Metadata table support (legacy path)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Today, metadata tables are mainly used by the ``visualize`` verb.

This path is expected to be reduced/deprecated over time. For new dataset code,
prefer explicit ``get_*`` methods for data you want to use in ML or visualization.

If you still need metadata-table behavior:

.. code-block:: python

   def __init__(self, config, data_location=None):
       metadata_table = ...
       super().__init__(config, metadata_table=metadata_table)


Complete minimal class
----------------------

.. code-block:: python

   from hyrax.datasets import HyraxDataset


   class MyDataset(HyraxDataset):
       def __init__(self, config, data_location=None):
           self.data_location = data_location
           self.flux_arrays = ...
           self.labels = ...
           super().__init__(config)

       def __len__(self):
           return len(self.flux_arrays)

       def get_flux(self, idx):
           return self.flux_arrays[idx].astype("float32")

       def get_label(self, idx):
           return int(self.labels[idx])

       def get_object_id(self, idx):
           return str(idx)


Streaming datasets
------------------

Everything above describes a *map-style* dataset: Hyrax knows how many objects
there are and can ask for any one of them by index. A live alert stream or a
catalog too large to index does not work that way, so Hyrax has a second, much
smaller contract for **streaming** datasets.

A streaming dataset inherits from ``torch.utils.data.IterableDataset`` in addition
to ``HyraxDataset``, and Hyrax routes it to the ``train_stream`` and
``infer_stream`` verbs instead of ``train``/``infer``:

.. code-block:: python

   import torch
   from hyrax.datasets import HyraxDataset


   class MyStreamDataset(HyraxDataset, torch.utils.data.IterableDataset):
       def __init__(self, config, data_location=None):
           ...
           super().__init__(config, metadata_table=None)

There are no ``get_<field>`` methods. Instead you implement four things:

* ``__iter__(self)`` - a generator yielding **batches**, each a ``list`` of flat
  ``dict``s mapping field name to value, for example
  ``[{"object_id": "abc", "flux": [...]}, ...]``. The dataset owns batching,
  because a ``DataLoader`` cannot emit a partial batch when data stops arriving.
* ``peek_sample(self)`` - return one flat sample ``dict`` **without losing it**.
  Hyrax uses this to build the model before training starts, and the sample must
  still show up in the first batch.
* ``stop(self)`` - signal ``__iter__`` to finish. Called when the session closes.
* ``__len__(self)`` - must exist, but should raise ``TypeError``; a stream has no
  length.

Pulling the object id out of each sample and grouping the model input fields is
handled for you by ``StreamingDataProvider``, driven by the same ``data_request``
``primary_id_field`` and ``fields`` keys used everywhere else. Your dataset just
emits flat dicts. ``collate_<field>`` hooks work exactly as they do above.

Streams always run with ``num_workers = 0``, because a plain ``IterableDataset``
does no sharding across worker processes - with several workers each would open
its own stream and every object would be seen more than once.

Two streaming datasets ship with Hyrax:
:class:`~hyrax.datasets.kafka_stream_dataset.KafkaStreamDataset` for live alert
streams, and :class:`~hyrax.datasets.lsdb_stream_dataset.LSDBStreamDataset` for
HATS catalogs read through LSDB. Read either one as a worked example.


Notebook-first path
-------------------

Start with :doc:`/pre_executed/external_dataset_class`, then move the class into
an external package when it works.
