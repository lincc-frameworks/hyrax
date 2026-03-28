Science Examples
================

These examples show how Hyrax can be used in realistic astronomy research
settings, where the goal is not just to run a model, but to inspect latent
spaces, identify unusual objects, and connect machine learning outputs back to
scientific interpretation.

.. tip::
   If you are new to Hyrax, we recommend you go to the
   :doc:`Getting Started <getting_started>` section before jumping into the science
   examples.
   

.. grid:: 1 1 2 2

   .. grid-item-card:: Extragalactic Unsupervised Discovery
       :link: pre_executed/unsupervised_hsc_full_pipeline
       :link-type: doc

       :bdg-primary:`HSC` :bdg-secondary:`Unsupervised` :bdg-success:`Galaxies`

       An end-to-end unsupervised discovery workflow on HSC galaxy cutouts.
       This notebook trains an autoencoder, builds a UMAP view of the latent
       space, explores that space interactively, runs nearest-neighbour search
       through a vector database, and uses distance in latent space to surface
       unusual objects for follow-up.

   .. grid-item-card:: Transient Classification from Light Curves
       :link: pre_executed/supervised_lightcurve_transients
       :link-type: doc

       :bdg-primary:`PLAsTiCC` :bdg-secondary:`Supervised` :bdg-info:`Transients`

       A supervised classification workflow on multi-band light curves from the
       PLAsTiCC dataset. This notebook defines a custom dataset and a custom 1-D
       CNN model, trains the classifier on 14 classes of astronomical transients
       and variables, and evaluates performance with a confusion matrix.

.. important::
   This section will continue to grow. Over time, we plan to add additional
   science examples covering different data types, surveys, and models. If you
   have an ML-oriented use case in mind that is not covered here, chances are
   Hyrax can support it. If you are not sure how to approach it, get in touch
   with us. We are here to help.
   

.. toctree::
   :hidden:

   Extragalactic Unsupervised Discovery <pre_executed/unsupervised_hsc_full_pipeline>
   Transient Classification from Light Curves <pre_executed/supervised_lightcurve_transients>
