.. _model_comparison:

Model comparison
================

One goal of Hyrax is to make model evaluation easier. Many tools exist for visualization
and evaluation of models. Hyrax integrates with TensorBoard and MLFlow to provide
easy access to these tools.

TensorBoard
-----------

Hyrax automatically logs training, validation and gpu metrics (when available) to
TensorBoard while training a model.
This allows for easy visualization of the training process.

For more information about TensorBoard see the
`TensorBoard documentation <https://www.tensorflow.org/tensorboard/get_started>`_.

MLFlow
------

Hyrax supports MLFlow for model tracking and experiment management.
By default the data collected for each run will be nested under the experiment
"notebook" using a run name that is the same as the results directory,
i.e. <timestamp>-train-<uid>.

The MLFlow server can be run from within a notebook or from the command line.

.. tabs::

    .. group-tab:: Notebook

        .. code-block:: python

           # Start the MLFlow UI server
           backend_store_uri = f"file://{Path(f.config['general']['results_dir']).resolve() / 'mlflow'}"
           mlflow_ui_process = subprocess.Popen(
               ["mlflow", "ui", "--backend-store-uri", backend_store_uri, "--port", "8080"],
               stdout=subprocess.PIPE,
               stderr=subprocess.PIPE,
           )

           # Display the MLFlow UI in an IFrame in the notebook
           IFrame(src="http://localhost:8080", width="100%", height=1000)

    .. group-tab:: CLI

        .. code-block:: bash

           >> mlflow ui --port 8080 --backend-store-uri <results_dir>/mlflow

        If you are running mlflow on a remote server, you will need to add the `--host` flag to the command:

        .. code-block:: bash
            
           >> mlflow ui --port 8080 --backend-store-uri <results_dir>/mlflow --host 0.0.0.0

        on the remote server, and the forward the port (8080) to your local machine using SSH.


For more information about MLFlow see the
`MLFlow documentation <https://mlflow.org/docs/latest/index.html>`_.
