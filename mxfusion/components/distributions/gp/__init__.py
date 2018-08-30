"""This module contains implementations of Gaussian Processes for MXFusion.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    gp
    cond_gp
    kernels
"""

__all__ = ['kernels', 'gp', 'cond_gp']

from .gp import GaussianProcess
from .cond_gp import ConditionalGaussianProcess
