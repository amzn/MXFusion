# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================


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

from .runtime_variable import add_sample_dimension, add_sample_dimension_to_arrays, expectation, array_has_samples, get_num_samples, as_samples
from .var_trans import Softplus, PositiveTransformation
from .variable import Variable, VariableType
