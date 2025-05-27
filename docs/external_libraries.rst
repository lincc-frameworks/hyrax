Defining custom Models and Datasets
===================================

Hyrax supports defining custom models and data sets in your notebook, or in a python package that you can share
with your community.

Two basic conditions must be met to use a custom model or dataset library:

#. The relevant class must be defined under the appropriate decorator or superclass. Models must be decorated with ``@hyrax_model`` and Datasets must inherit from ``HyraxDataset``.
#. The name of the class must be noted in the hyrax config. ``[model]`` ``name`` for models, or ``[data_set]`` ``name`` for data sets.

Configuring an external class
-----------------------------

The ``name`` configuration under either the ``[model]`` or ``[data_set]`` config sections is the dotte python 
name used to locate the class starting at the top package level. e.g. if your dataset class is called ``MyDataSet`` and 
is in a package called ``mydataset``, then you would configure as follows:

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

            from hyrax import Hyrax
            h = Hyrax()
            h.config["data_set"]["name"] = "mydataset.MyDataSet"

    .. group-tab:: CLI

        .. code-block:: bash

            $ cat hyrax_config.toml
            [data_set]
            name = "mydataset.MyDataSet"

Datasets in the current notebook, or within your own package can simply be referred to by their class names without any dots.

It is a valid usage of this extensibility to write your own dataset or model inline in the notebook where you 
are using Hyrax. Just be sure to re-run the cell where your model class is defined when you change it!

Defining a Model
----------------

Models must be written as a subclasses of ``torch.nn.Module``, use pytorch for computation, and 
be decorated with ``@hyrax_model``. Models must minimally define ``__init__``, ``forward``, and ``train_step`` 
methods.

In order to get the ``@hyrax_model`` decorator you can import it with ``from hyrax.models import hyrax_model``.

``__init__(self, config, shape)``
.................................
On creation of your model Hyrax passes the entire Hyrax config as a nested dictionry in the ``config`` argument
as well as the shape of each item in the dataset we intend to run on in the ``shape`` argument. This data is provided 
to allow your model class to adjust architecture or check that the provided dataset will work appropriately.

``shape`` is a tuple having the length of each individually iterable axis. An image dataset consisting of 
250x250 px images with 3 color channels each might have a shape of (3, 250, 250) indicating that the color channels are 
the first iterable axis of the tensor.


``forward(self, x)``
....................
Hyrax calls this function evaluates your model on a single input ``x``. ``x`` is gauranteed to be a tensor with 
the shape passed to ``__init__``. 

``forward()`` ought return a tensor with the output of your model.


``train_step(self, batch)``
...........................
This is called several times every training epoch with a batch of input tensors for your model, and is the 
inner training loop for your model. This is where you compute loss, perform back propagation, etc depending on 
how your model is trained.

``train_step`` returns a dictionary with a "loss" key who's value is a list of loss values for the individual 
items in the batch. This loss is logged to MLflow and tensorboard.

Optional Methods
................

``@staticmethod to_tensor(data_dict)``
......................................
This function is optional. It exists to allow model writers flexibility on how they present scientific data 
to their model, and to allow dataset authors to make datasets without constraining model authors to a 
particular ML architecture.

Defining ``to_tensor`` is necessary when a dataset returns a dictionary as the individual datum, rather than 
a ``Torch.tensor``.  ``to_tensor`` takes a batch of whatever is returned by the Dataset class, and returns 
a batch of ``Torch.tensor`` appropriate to send to the model's ``forward`` function

For example, we can consider a dataset that returns a dictionary of telescope data for a particular object. 
In our example ``flux_*`` are 2d images of calibrated fluxes, ``spectrum`` is a list of fluxes at different
frequencies, and ``mag_g`` is a magnitude for the g filter. The dataset would return items that look like the 
python dictionary below:

.. code-block:: python

    # What the dataset gives as a single item
    {
    "flux_g": <Torch Tensor>,
    "flux_r": <Torch Tensor>,
    "flux_i": <Torch Tensor>,
    "spectrum": <numpy.array>,
    "mag_g": <numpy.float32>,
    }

The model's ``to_tensor`` function will recieve a batch dictionary, where each key will have a list of the 
relevant data as shown below:

.. code-block:: python

    # What to_tensor recieves from hyrax
    {
    "flux_g": [ <Torch Tensor>, <Torch Tensor>, <Torch Tensor>, ...],
    "flux_r": [ <Torch Tensor>, <Torch Tensor>, <Torch Tensor>, ...],
    "flux_i": [ <Torch Tensor>, <Torch Tensor>, <Torch Tensor>, ...],
    "spectrum": [ <numpy.array>, <numpy.array>, <numpy.array>, ...],
    "mag_g": [ <numpy.float32>, <numpy.float32>, <numpy.float32>, ...],
    }

``to_tensor`` must return a list of ``Torch.tensor`` objects that your ``forward`` function can accept as 
it's ``x`` input. See the example below, which stacks the g, r, and i fluxes into a single tensor:

.. code-block:: python

    from hyrax.models import hyrax_model

    @hyrax_model
    class MyModel:

        @staticmethod
        def to_tensor(batch_dict):
            """
            Accepts a dictionary of tensor batches for individual telescope filters.
            Returns a batch of stacked tensor with the first index corresponding to the 
            filters g, r, and i respectively.
            """
            g_imgs = batch_dict["flux_g"]
            r_imgs = batch_dict["flux_r"]
            i_imgs = batch_dict["flux_i"]

            stacked_imgs = [
                torch.stack(g_img, r_img, i_img) 
                for g_img, r_img, i_img in zip(g_imgs, r_imgs, i_imgs)
            ]

            return stacked_images

Note that ``to_tensor`` must be defined with ``@staticmethod`` as in the example. The function does not have
access to the model's data members through the typical ``self`` argument in python.

Another possible use of ``to_tensor`` is to remove or otherwise adjust the input data of your model in ways 
that are not easily done with a ``torch.transform``. Below is an example ``to_tensor`` function which removes 
NaN values from input data, replacing them with the value zero. 

.. code-block:: python

    from hyrax.models import hyrax_model

    @hyrax_model
    class MyModel:

        @staticmethod
        def to_tensor(batch_dict):
            """
            Accepts a batch of tensors which may contain NaN values. Replaces those values with zero.
            """
            from torch import any, isnan, nan_to_num
            if any(isnan(batch)):
                batch = nan_to_num(batch, 0.0)
            return batch

Some NaN handling is available automatically in hyrax, via ``config['data_set']['nan_mode']``, but if a 
customized strategy is desired, the approach above may be preferable.

.. _custom-dataset-instructions:

Defining a dataset class
------------------------

Dataset classes are written as subclasses of ``hyrax.data_sets.HyraxDataset``. Datasets must choose to be 
either "map style", and also inherit from ``torch.utils.data.Dataset`` or "iterable" and inherit from 
``torch.utils.data.IterableDataset``. `Look here <https://pytorch.org/docs/stable/data.html#dataset-types>`_ 
for an overview of the difference between map style and iterable datasets.

A fully worked example of creating a custom map-style dataset class is in the example notebook 
:doc:`/pre_executed/custom_dataset` If you are writing a dataset for the first time, this is the best place 
to start.

When creating a dataset it is easiest to test it using the ``prepare`` verb to hyrax like so:

.. code-block:: python

    import hyrax
    h = hyrax.Hyrax()
    h.config["data_set"]["name"] = "<ClassNameOfYourDataset>"
    # Other config your dataset needs goes here

    dataset = h.prepare()
    dataset[0] # will get the first element for a map-style dataset
    next(iter(dataset)) # will get the first element for an iterable dataset
    len(dataset) # will return the length of your dataset
    list(dataset.ids()) # will list the ids in your dataset.

The dataset returned from ``prepare`` will be an instance of your class if running ``__init__`` did not 
cause an error. You can then do things like index your class or call the methods in your class to ensure
they are working as intended. 

The methods required are detailed below:

All datasets
............

``__init__(self, config)``
.................................
On creation of your dataset Hyrax passes the entire Hyrax config as a nested dictionry in the ``config`` 
argument. It is assumed that your dataset will handle the whole of your dataset, and any splitting of the 
dataset will be done by Hyrax, when running the relevant verb. Further detail on splitting can be found in 
:doc:`/data_set_splits`

You must call ``super().__init__(config)`` or ``super().__init__(config, metadata_table)`` in your 
``__init__`` function

Map style datasets
..................

``__getitem__(self, idx:int)``
..............................
Return a single item in your dataset given a zero-based index. This function may return either a 
``torch.Tensor`` or a dictionary of named data values that could be converted into a ``torch.Tensor`` by the
model's ``to_tensor`` method (see above).  

In situations where there is tight coupling between the model and data, or only one real way to pack the 
data into a tensor for ML applications, we recommend returning a ``torch.Tensor``.  If there are multiple ways
to pack the data, and it is primarily a question of model architecture, we recommend going the dictionary 
route.

In situations where a dataset's ``__getitem__`` returns a dictionary, and the model has not defined a 
``to_tensor`` function, Hyrax will use the ``"image"`` and ``"label"`` keys in the dictionary to give the 
model a tensor and an optional label. If these keys do not exist, Hyrax will prompt that a ``to_tensor`` 
function must be defined on the model before training or inference can proceed.

``__len__(self)``
.................
Return the length of your dataset.

Iterable datasets
.................

``__iter__(self)``
..................
Yield a single item in your dataset, or supply a generator function which does the same.
If your dataset has an end, yield ``StopIteration`` at the end.

Warning: Iterable datasets which never yield ``StopIteration`` are not currently supported in hyrax.

See the documentation on ``__getitem__`` regarding the value the generator ought yield.

Optional Overrides
..................

``ids(self)``
.............
Return a list of IDs for the objects in your dataset. These IDs ought be returned as a string generator that 
yields the ids in the order of your dataset.

