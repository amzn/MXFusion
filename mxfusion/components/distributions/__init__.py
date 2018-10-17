"""This module contains Distributions for MXFusion.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    categorical
    distribution
    normal
    gamma
    pointmass
    random_gen
    univariate
    gp
    wishart
    beta
"""

__all__ = ['categorical', 'distribution', 'normal', 'gamma', 'pointmass', 'random_gen',
           'univariate', 'gp', 'wishart', 'beta']

from .distribution import Distribution
from .normal import Normal, MultivariateNormal, NormalMeanPrecision, MultivariateNormalMeanPrecision
from .gamma import Gamma, GammaMeanVariance
from .pointmass import PointMass
from .categorical import Categorical
from .gp import GaussianProcess, ConditionalGaussianProcess
from .wishart import Wishart
from .beta import Beta
