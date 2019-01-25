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


"""This module contains implementations of Gaussian Processes for MXFusion.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    add_kernel
    kernel
    linear
    matern
    rbf
    static
    stationary
"""

__all__ = ['add_kernel', 'kernel', 'linear',  'matern', 'rbf', 'static',
           'stationary']

from .add_kernel import AddKernel
from .rbf import RBF
from .linear import Linear
from .static import Bias, White
from .matern import Matern52, Matern32, Matern12
