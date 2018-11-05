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


"""The main module for MXFusion.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    components
    inference
    models
    modules
    util
"""

__all__ = ['components', 'models', 'modules', 'inference', 'util']

from .components import distributions, functions, variables
from .models import Model, Posterior
from .components import Variable

from .__version__ import __version__
