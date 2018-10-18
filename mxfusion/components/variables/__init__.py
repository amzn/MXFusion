"""Contains the Variable class, Variable transformations, and runtime methods on variables.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    runtime_variable
    var_trans
    variable
"""

__all__ = ['runtime_variable', 'var_trans', 'variable']

from .runtime_variable import add_sample_dimension, add_sample_dimension_to_arrays, expectation, is_sampled_array, get_num_samples, as_samples
from .var_trans import Softplus, PositiveTransformation
from .variable import Variable, VariableType
