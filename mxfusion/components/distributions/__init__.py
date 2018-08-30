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
"""

__all__ = ['categorical', 'distribution', 'normal', 'pointmass', 'rand_gen',
           'univariate','gp']

from .distribution import Distribution
from .normal import Normal, MultivariateNormal
from .pointmass import PointMass
from .categorical import Categorical
from .gp import GaussianProcess, ConditionalGaussianProcess
