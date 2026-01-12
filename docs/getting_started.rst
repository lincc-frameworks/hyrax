Getting started with Hyrax
==========================


Installation
-------------
Hyrax can be installed via pip:

.. code-block:: bash

   pip install hyrax

Hyrax is officially supported and tested with Python versions 3.10, 3.11, 3.12, and 3.13.
Other versions may work but are not guaranteed to be compatible.

We strongly encourage the use of a virtual environment when working with Hyrax
because Hyrax depends on several open source pacakges that may have conflicting
dependencies with other packages you have installed.


First Steps
-----------

Some kind of preamble that talks about the CiFAR10 dataset and what we're going to do.
Similar to the classic PyTorch CIFAR example:
https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-an-image-classifier

Create a hyrax instance
~~~~~~~~~~~~~~~~~~~~~~~

The main driver for Hyrax is the ``Hyrax`` class. To get started we'll create an
instance of this class.

.. code-block:: python

   from hyrax import Hyrax

   h = Hyrax()

When we create the Hyrax instance, it will automatically load a default configuration
file. This file contains default settings for all of the components that Hyrax uses.

Specify a model
~~~~~~~~~~~~~~~

We'll need to let Hyrax know which model to use for training.
Here we'll tell Hyrax to use the built-in HyraxCNN model that is based on the
<simple CNN architecture|https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network>
from the PyTorch CIFAR10 tutorial.

.. code-block:: python

   h.set_config('model.name', 'HyraxCNN')


Defining the dataset
~~~~~~~~~~~~~~~~~~~~~~

We'll also need to tell Hyrax what data should be used for training, in this case
the CIFAR10 dataset.
Hyrax has a built in dataset class for working with CIFAR10 data, so we'll configure
that here.
You can learn more about the CIFAR10 at the offical site:
https://www.cs.toronto.edu/~kriz/cifar.html

.. code-block:: python
   :linenos:

   model_inputs_definition = {
        "train": {
            "data": {
                "dataset_class": "HyraxCifarDataset",
                "data_location": "./data",
                "fields": ["image", "label"],
                "primary_field_id": "object_id",
            },
        }
    }

    h.set_config("model_inputs", model_inputs_definition)

This may appear overwhelming, especially for a simple case, but being explicit
about the dataset configuration will allow for great flexibility down the line
when working with more complex data.

Training the model
~~~~~~~~~~~~~~~~~~

Now that we have the model and data specified, we're ready for training.
We'll use the ``train`` verb to kick off the training process.

.. code-block:: python

   h.train()

Once the training is complete, the model weights will be saved in a timestamped
directory with a name similar to ``/YYYYmmdd-HHMMSS-train-xxxx``.


Testing the model
~~~~~~~~~~~~~~~~~~

Now that we've trained a model, we can evaluate its performance on the test dataset.
First we'll add to our model input definition to specify the data to use for inference.

.. code-block:: python
   :linenos:

   model_inputs_definition["infer"] = {
       "data": {
           "dataset_class": "HyraxCifarDataset",
           "data_location": "./data",
           "fields": ["image"],
           "primary_field_id": "object_id",
           "dataset_config": {
               "use_training_data": False,
           },
       },
   }

   h.set_config("model_inputs", model_inputs_definition)

Then we'll use Hyrax's ``infer`` verb to load the trained model weights and process
the data defined above.

.. code-block:: python

   h.infer()

