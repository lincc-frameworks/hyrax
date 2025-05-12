"""
This is a modified copy of the Hyper Suprime-Cam (HSC) download cutout tool published for pdr3 on the
`HSC Gitlab repository <https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/tree/master/pdr3/downloadCutout>`_

It is tightly integrated with the ``download`` verb in hyrax and has been modified so as to allow:
#. Better error reporting and recovery
#. Multithreaded use by the ``download`` hyrax verb
#. Creation and updating of download manifests when run by the download verb.

"""

# ruff: noqa: F403
from .downloadCutout import *

__all__ = []
