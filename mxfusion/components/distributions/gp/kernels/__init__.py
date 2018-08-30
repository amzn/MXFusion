"""This module contains implementations of Gaussian Processes for MXFusion.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    add_kernel
    kernel
    linear
    rbf
    static
    stationary
"""

__all__ = ['add_kernel', 'kernel', 'linear', 'rbf', 'static', 'stationary']

from .add_kernel import AddKernel
from .rbf import RBF
from .linear import Linear
from .static import Bias, White
