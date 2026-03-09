# Hyrax
### A low-code solution for rapid experimentation with machine learning in astronomy
[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/hyrax/smoke-test.yml)](https://github.com/lincc-frameworks/hyrax/actions/workflows/smoke-test.yml)
[![codecov](https://codecov.io/gh/lincc-frameworks/hyrax/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks/hyrax)
[![Read the Docs](https://img.shields.io/readthedocs/hyrax)](https://hyrax.readthedocs.io/en/latest)
[![PyPI](https://img.shields.io/pypi/v/hyrax?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/hyrax/)

Hyrax is an extensible platform that handles much of the boilerplate code that is often required for a machine learning project in astronomy. Hyrax users are able to focus on the science work of model development and results analysis instead of infrastructure.

Hyrax is not tied to a specific model or data modality but rather is intended to encourage an ecosystem of models and data for rapid experimentation.
If the algorithm you want can be implemented in PyTorch, then Hyrax can likely reduce the boilerplate code required for a reproducible project.


## Getting Started 
Hyrax can be installed via pip:

```
>> pip install hyrax
```

Hyrax is officially supported and tested with Python versions 3.11, 3.12, and 3.13.
Other versions may work but are not guaranteed to be compatible.

Check out [Getting started](https://hyrax.readthedocs.io/en/latest/getting_started.html) and
[Common workflows](https://hyrax.readthedocs.io/en/latest/common_workflows.html) in the documentation for usage examples.


## Existing Hyrax Projects
Hyrax has been developed to support single and multimodal data for use with both supervised and unsupervised models.
Some examples include: 

- Image-based unsupervised discovery in Rubin-LSST and HSC. (A. Ghosh, J.  Chatchadanoraset, D. Miura)
- Spectra-based supervised clustering to study supernova Ia spectral diversity. (L. Cunningham, M. Dai)
- Image-based supervised small body classification. (M. West++)
- Multimodal time-series classification for ZTF alert follow-up. (A. Sasli, F. Fontinele-Nunes++)
- Image-based unsupervised discovery of cluster-scale gravitationally lensed arcs. (G. Khullar++)
- Searches for semi-resolved galaxies in HSC and LSST (P. Ferguson ++)

## Collaborations and Contributions
If you are an astronomer interested in using Hyrax, please get in touch with us!
We are especially interested to hear about applications that Hyrax doesn't currently support.

Hyrax is open source and under active development.
If you would like to contribute, please contact us. We would be happy to work with you.


## Acknowledgements
This project started as a collaboration between different units within the
[LSST Discovery Alliance](https://lsstdiscoveryalliance.org/) --
the [LINCC Frameworks Team](https://lsstdiscoveryalliance.org/programs/lincc-frameworks/)
and LSST-DA Catalyst Fellow, [Aritra Ghosh](https://ghosharitra.com/).

This project is supported by Schmidt Sciences and the John Templeton Foundation
