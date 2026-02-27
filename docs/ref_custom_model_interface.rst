
Custom Models
=============

Required Interface
------------------

Models must be written as a subclasses of ``torch.nn.Module``, use pytorch for computation, and 
be decorated with ``@hyrax_model``. Models must also minimally define ``__init__``, ``forward``, 
and ``train_step`` methods outlined below

In order to get the ``@hyrax_model`` decorator you can import it with ``from hyrax.models import hyrax_model``
wherever you define your model class.

``__init__(self, config, data_sample)``
.......................................
On creation of your model Hyrax passes the fully resolved Hyrax runtime config as a nested dictionary in the 
``config`` argument. If your model is part of a package that takes configuration variables, or if you've set 
custom config parameters for your model otherwise, those configuration parameters will be part of the nested 
``config`` dictionary. 

The ``config`` dictionary is intended to be read-only. Alterations to the config dictionary made in this 
function should not directly affect behavior elsewhere in hyrax or your own custom classes. 

Hyrax also passes in a single data sample, so that things like array size mismatches or
data type issues can be discovered during ``__init__``. Additionally, if your model works differently on 
different data, you can adjust the creation of the neural net architecture from this sample data as needed.

``forward(self, batch)``
........................
This is the main method that evaluates your model. ``batch`` will consistently be a pytorch tensor, where 
the first index is the configured batch size, and the remaining indexes have the structure and data type(s)
of the ``data_sample`` passed to ``__init__``.

``forward()`` ought return a tensor that is the output of your model, with the same batch multiplicity and 
order in the first tensor index as the ``batch`` that was passed in.


``train_step(self, batch)``
...........................
This is called for each batch of data during training, and is where you define the training loop for your 
model. This function should compute loss, perform back propagation, etc depending on how your model is 
trained.

``train_step`` returns a dictionary with a "loss" key who's value is a list of loss values for the individual 
items in the batch. This loss is logged to MLflow and tensorboard, as are any other values returned in the
dictionary.


``@staticmethod to_tensor(data_dict)``
......................................
This function defines the mapping from your dataset's fields to the actual packed tensor that your model 
will ultimately process. For reference information on defining the input data to this function see xcxc 
configuring data

``data_dict`` is a dictionary, where each value is a list of all the batched values corresponding to that 
key, laid out like the example below. 

.. code-block:: python

    data_dict = {
        "data": {
            "flux_g": [ <numpy.array>, <numpy.array>, <numpy.array>, ...],
            "flux_r": [ <numpy.array>, <numpy.array>, <numpy.array>, ...],
            "flux_i": [ <numpy.array>, <numpy.array>, <numpy.array>, ...],
            "spectrum": [ <numpy.array>, <numpy.array>, <numpy.array>, ...],
            "mag_g": [ <numpy.float32>, <numpy.float32>, <numpy.float32>, ...],
            "object_id": [ <numpy.int64>, <numpy.int64>, <numpy.int64>, ... ],
        }
        "object_id": [ <numpy.int64>, <numpy.int64>, <numpy.int64>, ... ],
    }

In the case of multimodal data, the multiple datasets will be in their own separate named dictionaries within 
``data_dict``.  If you have defined a ``collate_fn`` either on your dataset class, or configured one in the
global hyrax configuration, the internal structure of the sub-dictionaries will match the outputs of 
``collate_fn``

``to_tensor`` must output a numpy ndarray, where the first index indexes within the batch, and other indexes 
are as your model expects to see in ``forward`` and ``__init__``.

For the above example if we had a machine learning model that expected to see a multi-layer image tensor with 
the g, r, and, i band flux data, you would write the following function:

.. code-block:: python

    @staticmethod
    to_tensor(data_dict):
        import numpy as np 

        # Each of 'flux_*' have x/y pixel as indexes
        g = np.array(data_dict['data']['flux_g'])
        r = np.array(data_dict['data']['flux_r'])
        i = np.array(data_dict['data']['flux_i'])

        # Stack g,r,i
        gri = np.array([g,r,i])

        # Move batch axis to the front
        return np.moveaxis(gri, 1, 0)

It is important that all imports you need to run ``to_tensor`` are imported from inside the function body as 
shown above. This is unfortunately required for onnx export of the model to work, and is assumed by some 
verbs.

Optionally Configurable Training
--------------------------------

Hyrax offers several automatic member variables to your class. You can access these variables in class 
methods, but they are initialized for your class by the framework. The purpose is to expose various pluggable 
aspects of machine learning training to hyrax configuration so that initializing many different types of 
training can be scripted in an HPC or batch context by only changing configuration, not the code of the model.

``self.criterion``
..................

In this variable Hyrax provides a ``torch.nn`` which can be used as a loss function. The 
class name of the criterion is controlled by ``config["criterion"]["name"]``, and the default is 
``torch.nn.CrosEntropyLoss``.  Other configuration variables under criterion are used as keyword arguments
to the constructor. Note that Hyrax makes this varible available; however, it is only used in training if 
explicitly referenced in the ``train_step`` method

``self.optimizer``
..................

In this variable Hyrax provides a torch optimizer, which can be used to train your model.
The class name of the optimizer is contrlled by ``config["optimizer"]["name"]``, and the default is 
``torch.optim.SGD``. This optimizer is also only used for training if explicitly referenced in the 
``train_step`` method

Best Practices
--------------

Preflight your model in ``__init__``, by running the model on ``sample_data``. It is far 
easier to debug mismatches between model and data in ``__init__`` than during model training. Model/data 
mismatches are the most common cause of errors during the initial setup of a training run.

Return key values that will tell you about training success or model quality in ``train_step``, so that they
show up in your tensorboard and MLflow dashboards. This allows you to see how your model is changing during 
training and extract insights which can help you get to the highest quality model.

Prefer configuring hyperparameters (or other variations in training) over hardcoding. By allowing 
hyperparameters to your model, or small variations in how it is trained to be configured you make running 
a grid of hyperparameter values a matter of running Hyrax several times with slightly different 
configurations. It is possible to automate this process using frameworks like Optuna.

.. insert optuna training example

.. Use version control for your model classes?

