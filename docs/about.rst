About Hyrax
===========

What is Hyrax?
--------------
Hyrax is a powerful and extensible machine learning framework that automates data
acquisition, scales seamlessly from laptops to HPC, and ensures reproducibility 
— freeing astronomers to focus on discovery instead of infrastructure.

Put another way, it's an effort to bring the best practices of software engineering
to the astronomy machine learning community.


Why build Hyrax?
----------------
Image-based machine learning in astronomy is challenging work.
It requires data collection, pre-processing, model training and evaluation, and
analysis of results of inference - all of which introduce potential bottlenecks
for new science.

We've found that many bottlenecks require significant effort to overcome, and that
effort doesn't accrue to science, it's just a means to an end.
And worse, it's often duplicated effort by many different people, each solving
the same problems in slightly different ways.

Hyrax is our effort to make the process easier and more efficient by taking care of
the common tasks so that scientists can focus on the science.


Guiding principles
------------------
* **Principle 1** Empower the scientists to do science - not software engineering.
  Hyrax automatically discovers and uses the most performant hardware available
  for training without any changes to the users code.
* **Principle 2** Make the software easy to use.
  Hyrax is designed to be used in a Jupyter notebook for exploration or from the
  command line within HPC or Slurm environments without modification.
* **Principle 3** Make the software extensible to support many different use cases.
  We work closely with scientists to build Hyrax to support their use cases, but
  we learned early on that we can't anticipate all the ways that Hyrax will be used.
  Hyrax is designed to be easily extended to support new models, data sources,
  and functionality (see :doc:`/external_libraries`).


Commitment to open science
---------------------------
Hyrax is open source software, and makes extensive use of open source libraries.
We envision a Hyrax ecosystem where users can freely share data, trained models,
and other components to accelerate their research and the research of others.
