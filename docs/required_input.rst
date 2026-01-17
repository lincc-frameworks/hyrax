Required inputs
===============

Hyrax covers a lot of the "plumbing" required to run machine learning experiments,
however there are still a few key places where a user needs to provide input for
Hyrax to know what to do.

Default values are provided for almost all configurable aspects of Hyrax, but
there are three essential pieces of information that a user must provide:

#. The model to use for training or inference
#. The data that will be provided to the model
#. The function needed to prepare the data for the model

.. admonition:: Hyrax design choice

   Why do I need to provide data *and* tell Hyrax how to prepare the data for
   the model? That seems like extra work!

   In Hyrax we want to support rapid experimentation with different models and datasets.
   To do this, we separate the data from the model.
   When we separate these, a *dataset* can be used with multiple models,
   and a *model* can be used with multiple datasets.

   This increases reusability and allows users to quickly swap out datasets and models
   without needing to make changes to the code in either.

Instructions for Hyrax come in two forms - simple configuration settings and user-defined code.


Define the model
------------------
At it's core, Hyrax is a framework for training and running machine learning models.
To do this, a user must provide the model that Hyrax will use for training or inference!

There's no reasonable default model that Hyrax could provide, so this is an essential piece
of information that a user must provide.


Define the data
-----------------
Hyrax needs to know what data to provide to the model during training or inference.
To do this, a user must provide a dataset that Hyrax can use to load data samples.

Again, there's no reasonable default dataset that Hyrax could provide, it's use-case
specific, so this is another essential piece of information that a user must provide.


Prepare the data
----------------
To support reusablility, Hyrax encourages the separation of datasets and models.
i.e. the form of the data provided by a dataset should not be customized for a
specific model, nor should a model architecture be hardcoded to a specific dataset.

In order to convert the output of a dataset into a form that is suitable for a model,
Hyrax requires a user-defined function that will prepare the data for the model.

As a contrived example, consider a dataset where each data point is a composed of
three 2d arrays: "science", "mask", and "variance".
To be flexible, the dataset class that provides data from disk would return each
array individually.
The model may expect that a single flattened array as input.
If the model accepted a single flattened array as input, the user would need to
define a `to_tensor` function that takes in the structured data sample and stack
the arrays into a single array before returning it.

The user may then want to quickly experiment with the model by first providing
only the "science" array as input, and later adding in the "variance" array as well.
By defining a custom `to_tensor` function, the user can easily modify how the
data is prepared for the model without needing to modify either the dataset or
the model code.
