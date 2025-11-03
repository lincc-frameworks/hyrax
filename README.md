# Hyrax
[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/hyrax/smoke-test.yml)](https://github.com/lincc-frameworks/hyrax/actions/workflows/smoke-test.yml)
[![codecov](https://codecov.io/gh/lincc-frameworks/hyrax/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks/hyrax)
[![Read the Docs](https://img.shields.io/readthedocs/hyrax)](https://hyrax.readthedocs.io/en/latest)
[![PyPI](https://img.shields.io/pypi/v/hyrax?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/hyrax/)
<hr>

## Introduction
Hyrax is an efficient tool
to hunt for rare and anomalous sources in large astronomical imaging surveys
(e.g., Rubin-LSST, HSC, Euclid, NGRST, etc.).
Hyrax is designed to support four primary steps in the anomaly detection workflow:

* Downloading large numbers of cutouts from public data repositories
* Building lower dimensional representations of downloaded images -- the latent space
* Interactive visualization and algorithmic exploration (e.g., clustering, similarity-search, etc.) of the latent space
* Identification & rank-ordering of potential anomalous objects

Hyrax is not tied to a specific anomaly detection algorithm/model or a specific
class of rare/anomalous objects; but rather intended to support any algorithm
that the user may want to apply on imaging data.
If the algorithm you want to use takes in tensors, outputs tensors, and can be
implemented in PyTorch; then chances are Hyrax is the right tool for you!

## Getting Started 
To get started with Hyrax, clone the repository and create a new virtual environment.
If you plan to develop code, run the ``.setup_dev.sh`` script.

```
>> git clone https://github.com/lincc-frameworks/hyrax.git
>> conda create -n hyrax python=3.10
>> bash .setup_dev.sh (Optional, for developers)
```

## Additional Information
Hyrax is under active development and has limited documentation at the moment.
We aim to have v1 stability and more documentation in the first half of 2025.
If you are an astronomer trying to use Hyrax before then, please get in touch with us!

This project started as a collaboration between different units within the
[LSST Discovery Alliance](https://lsstdiscoveryalliance.org/) --
the [LINCC Frameworks Team](https://lsstdiscoveryalliance.org/programs/lincc-frameworks/)
and LSST-DA Catalyst Fellow, [Aritra Ghosh](https://ghosharitra.com/).

## Acknowledgements

This project is supported by Schmidt Sciences.
