"""The main module for MXFusion.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    factor_graph
    model
    posterior
"""

__all__ = ['factor_graph', 'model', 'posterior']

from .model import Model
from .factor_graph import FactorGraph
from .posterior import Posterior
