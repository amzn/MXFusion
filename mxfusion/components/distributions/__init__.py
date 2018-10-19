"""This module contains Distributions for MXFusion.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    categorical
    distribution
    normal
    pointmass
    random_gen
    univariate
    gp
    laplace
"""

__all__ = ['categorical', 'distribution', 'normal', 'pointmass', 'random_gen',
           'univariate', 'gp', 'wishart', 'beta', 'laplace']

from .distribution import Distribution
from .gamma import Gamma, GammaMeanVariance
from .normal import Normal, MultivariateNormal
from .pointmass import PointMass
from .categorical import Categorical
from .gp import GaussianProcess, ConditionalGaussianProcess
from .wishart import Wishart
from .beta import Beta
from .laplace import Laplace
