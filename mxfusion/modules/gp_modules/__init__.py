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


"""This module contains Gaussian process modules.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    gp_regression
    sparsegp_regression
    svgp_regression
"""

__all__ = ['gp_regression', 'sparsegp_regression', 'svgp_regression']

from .gp_regression import GPRegression
from .sparsegp_regression import SparseGPRegression
from .svgp_regression import SVGPRegression
