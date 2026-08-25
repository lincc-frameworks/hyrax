Model class reference
=====================

This page is the ground truth for writing a Hyrax model class.

If you are not used to writing classes, follow this page method-by-method and
copy/edit the examples.


How Hyrax uses your model class
-------------------------------

Hyrax does this sequence:

1. Builds one sample batch from your dataset.
2. Calls ``YourModel.prepare_inputs(sample_batch)``.
3. Creates your model with ``YourModel(config, data_sample=prepared_sample)``.
4. During runs, calls:

   * ``train_batch`` for training
   * ``infer_batch`` for inference
   * ``validate_batch`` for validation during training (if you use ``validate`` data)
   * ``test_batch`` for test (if you run the ``test`` verb)


Required methods (checklist)
----------------------------

Your class must have these:

1. Inherit from ``torch.nn.Module``.
2. Add ``@hyrax_model`` above the class (this registers your model so Hyrax can
   find it by name in the :doc:`configuration </configuration>`).
3. Implement:

   * ``__init__(self, config, data_sample=None)``
   * ``forward(self, ...)`` (required by PyTorch modules)
   * ``infer_batch(self, batch)``
   * ``train_batch(self, batch)``

4. Set ``model.name`` to full class path in config.


Method-by-method guide
----------------------

``__init__(self, config, data_sample=None)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

What to do in this method:

1. Save ``config`` to ``self.config``.
2. Read model-specific settings from config.
3. Build your layers.
4. Optionally use ``data_sample`` to infer shapes.

The model-specific config values are typically defined in your external library
config layout. See :doc:`/external_library_package` for packaging/config setup.

Example (read settings from config):

.. code-block:: python

   def __init__(self, config, data_sample=None):
       super().__init__()
       self.config = config

       hidden = config["my_package"]["MyModel"]["hidden_size"]
       out_dim = config["my_package"]["MyModel"]["num_classes"]

       in_dim = 128
       if data_sample is not None:
           x, _ = data_sample
           in_dim = x.shape[-1]

       self.fc1 = nn.Linear(in_dim, hidden)
       self.fc2 = nn.Linear(hidden, out_dim)

To set those config values in a notebook, one simple pattern is:

.. code-block:: python

   h.config.setdefault("my_package", {}).setdefault("MyModel", {})["hidden_size"] = 64
   h.config["my_package"]["MyModel"]["num_classes"] = 10

PyTorch reference for building ``nn.Module`` layers:
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_


``forward(self, ...)``
^^^^^^^^^^^^^^^^^^^^^^

Even though Hyrax calls ``train_batch`` / ``infer_batch``, your model should still
implement ``forward`` because that is the core PyTorch pattern.

PyTorch reference for ``forward``:
`Module.forward <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward>`_

Example:

.. code-block:: python

   def forward(self, x):
       x = self.fc1(x)
       x = nn.functional.relu(x)
       x = self.fc2(x)
       return x


``@staticmethod prepare_inputs(data_dict)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This converts collated dataset fields into model inputs. For background on where
this sits in the pipeline, see :doc:`/data_flow`.

Important:

* define it as ``@staticmethod``
* return either:
  * one NumPy array, or
  * a tuple like ``(features, labels)``

Example:

.. code-block:: python

   @staticmethod
   def prepare_inputs(data_dict):
       features = data_dict["data"]["flux"]
       labels = data_dict["data"].get("label", None)
       return features, labels

If you do not define this method, Hyrax uses a default expecting
``data_dict["data"]["image"]`` and optional ``data_dict["data"]["label"]``.


``infer_batch(self, batch)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by the ``infer`` verb.

Usually this unpacks the batch and calls ``forward``.

Example:

.. code-block:: python

   def infer_batch(self, batch):
       features, _ = batch
       return self.forward(features)


``train_batch(self, batch)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by the ``train`` verb.

Typical steps:

1. unpack inputs
2. zero gradients
3. call ``forward``
4. compute loss
5. backprop + optimizer step
6. return a dictionary with ``"loss"`` and optional metrics

Example:

.. code-block:: python

   def train_batch(self, batch):
       features, labels = batch
       self.optimizer.zero_grad()
       logits = self.forward(features)
       loss = self.criterion(logits, labels)
       loss.backward()
       self.optimizer.step()
       return {"loss": loss.item(), "acc": 0.84}

Important details:

* Hyrax treats **lower ``loss`` as better** for best-checkpoint selection.
* Any extra metrics you return (like ``acc``) are logged to TensorBoard and MLflow.
  See :doc:`/notebooks/using_tensorboard_and_mlflow`. For documentation on comparing
  runs across experiments, see :doc:`model_comparison`.


Optional methods
----------------

``validate_batch(self, batch)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the training run includes a ``validate`` dataset group, the ``train`` verb will
run this method each epoch.

Recommended behavior: similar to ``train_batch`` but **do not** update weights.

Example:

.. code-block:: python

   def validate_batch(self, batch):
       features, labels = batch
       logits = self.forward(features)
       loss = self.criterion(logits, labels)
       return {"loss": loss.item(), "val_acc": 0.81}


``test_batch(self, batch)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used by the ``test`` verb.

Return a dictionary with ``"loss"`` (plus optional metrics).

Example:

.. code-block:: python

   def test_batch(self, batch):
       features, labels = batch
       logits = self.forward(features)
       loss = self.criterion(logits, labels)
       return {"loss": loss.item(), "test_acc": 0.82}


``log_epoch_metrics(self)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optional helper for extra per-epoch metrics during training.

If present, Hyrax calls it at epoch end and logs returned values to TensorBoard
and MLflow. See :doc:`/notebooks/using_tensorboard_and_mlflow`.

Example:

.. code-block:: python

   def log_epoch_metrics(self):
       return {"my_metric": 0.42}


``train_post_epoch(self)`` and ``validate_post_epoch(self)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optional zero-argument hooks called once at the end of each training epoch and
each validation epoch respectively. Unlike ``log_epoch_metrics``, they return
nothing -- use them to do work, such as writing out data you accumulated across
the epoch. They pair naturally with the run context described below.

.. note::

   These hooks are not supported for distributed training. If they are present
   and the world size is greater than one, Hyrax raises ``NotImplementedError``.

Example:

.. code-block:: python

   def train_post_epoch(self):
       self.flush_activations()


Optimizer / criterion / scheduler setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You have two ways to provide each training component:

* **Manual** means you create it in your model class (for example,
  ``self.optimizer = torch.optim.Adam(...)``).
* **Automatic** means you set its name in config and Hyrax creates it for you.

These choices are independent. You can mix them.

Example: manual optimizer + automatic criterion/scheduler

.. code-block:: python

   # in model __init__ (manual optimizer)
   self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

   # in config (automatic criterion + scheduler)
   h.set_config("criterion.name", "torch.nn.CrossEntropyLoss")
   h.set_config("scheduler.name", "torch.optim.lr_scheduler.StepLR")

This mixed mode is valid. Optimizer/criterion/scheduler do not all need to use
exactly the same style.


The run context
---------------

Scalar metrics returned from ``train_batch`` and friends go to TensorBoard and
MLflow. When you want to save something that does not fit that shape -- attention
maps, per-epoch embeddings, a JSON of diagnostics -- write it into the run's
results directory. ``get_context()`` tells you where that is.

.. code-block:: python

   from hyrax import get_context

   context = get_context()
   context["results_dir"]   # Path to this run's results directory

The context is a plain dictionary with these keys:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Meaning
   * - ``results_dir``
     - ``Path``
     - The results directory for the current run.
   * - ``verb``
     - ``str``
     - Which verb is running: ``"train"``, ``"infer"``, ``"test"``, etc.
   * - ``rank``
     - ``int``
     - The distributed rank of this process. ``0`` when not distributed.
   * - ``world_size``
     - ``int``
     - The number of distributed processes. ``1`` when not distributed.

Call ``get_context()`` wherever you like, including in ``__init__``. It always
returns the same dictionary object, and Hyrax fills that object in place when a
verb starts, so you never have to worry about initialization order.

.. code-block:: python

   from hyrax import get_context


   @hyrax_model
   class MyModel(nn.Module):
       def __init__(self, config, data_sample=None):
           super().__init__()
           self.config = config
           self.context = get_context()
           self.activations = []

       def train_batch(self, batch):
           features, labels = batch
           ...
           self.activations.append(hidden.detach().cpu().numpy())
           return {"loss": loss.item()}

       def train_post_epoch(self):
           rank = self.context["rank"]
           outfile = self.context["results_dir"] / f"activations_rank{rank}.npy"
           np.save(outfile, np.concatenate(self.activations))
           self.activations.clear()

Note the use of ``rank`` in the filename. Under distributed training every rank
shares one results directory, so include the rank in any filename you write or
the ranks will overwrite each other.

You may also put your own keys in the context. Hyrax clears it at the start of
each verb run, so anything you add lasts for that run only.

.. note::

   The context describes the run that is happening *now*. If you hold a model
   across two calls -- ``h.train()`` then ``h.infer()`` -- its context will point
   at the inference results directory during the second call. A model built
   outside of any verb run has an empty context, and reading a key from it raises
   a ``KeyError`` explaining why.


Complete minimal class
----------------------

.. code-block:: python

   import torch.nn as nn
   from hyrax.models.model_registry import hyrax_model


   @hyrax_model
   class MyModel(nn.Module):
       def __init__(self, config, data_sample=None):
           super().__init__()
           self.config = config
           self.fc = nn.Linear(128, 10)

       def forward(self, x):
           return self.fc(x)

       @staticmethod
       def prepare_inputs(data_dict):
           features = data_dict["data"]["flux"]
           labels = data_dict["data"].get("label", None)
           return features, labels

       def infer_batch(self, batch):
           features, _ = batch
           return self.forward(features)

       def train_batch(self, batch):
           features, labels = batch
           self.optimizer.zero_grad()
           out = self.forward(features)
           loss = self.criterion(out, labels)
           loss.backward()
           self.optimizer.step()
           return {"loss": loss.item()}


Notebook-first path
-------------------

Start with :doc:`/pre_executed/external_model_class`, then move the class into
an external package when it works.
