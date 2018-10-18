
"""This module contains functionality for using MXNet native operators in MXFusion models.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    operators
    operator_impl
"""

from .operators import Operator, MXNetOperatorDecorator
from .operator_impl import *
