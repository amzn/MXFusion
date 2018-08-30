"""The main module for MXFusion.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    components
    models
    inference
    util
"""

__all__ = ['components', 'models', 'inference', 'util']

from .components import distributions, functions, modules, variables
from .models import Model, Posterior
from .components import Variable
