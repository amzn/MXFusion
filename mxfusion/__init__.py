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

__all__ = ['components', 'models', 'modules', 'inference', 'util']

from .components import distributions, functions, variables
from .models import Model, Posterior
from .components import Variable

from .__version__ import __version__
