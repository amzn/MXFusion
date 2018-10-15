"""This module contains Gaussian process modules.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

"""

__all__ = ['gp_regression', 'sparsegp_regression', 'svgp_regression']

from .gp_regression import GPRegression
from .sparsegp_regression import SparseGPRegression
from .svgp_regression import SVGPRegression
