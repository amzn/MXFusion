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


"""This module contains Distributions for MXFusion.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    bernoulli
    categorical
    distribution
    normal
    gamma
    pointmass
    random_gen
    univariate
    gp
    laplace
    wishart
    beta
    dirichlet
"""

__all__ = ['bernoulli', 'categorical', 'distribution', 'normal', 'gamma', 'pointmass', 'random_gen',
           'univariate', 'gp', 'wishart', 'beta', 'laplace', 'uniform', 'dirichlet']

from .distribution import Distribution
from .normal import Normal, MultivariateNormal, NormalMeanPrecision, MultivariateNormalMeanPrecision
from .gamma import Gamma, GammaMeanVariance
from .pointmass import PointMass
from .bernoulli import Bernoulli
from .categorical import Categorical
from .gp import GaussianProcess, ConditionalGaussianProcess
from .wishart import Wishart
from .beta import Beta
from .dirichlet import Dirichlet
from .uniform import Uniform
from .laplace import Laplace
