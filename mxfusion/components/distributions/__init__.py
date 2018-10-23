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
