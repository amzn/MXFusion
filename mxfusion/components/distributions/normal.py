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


import numpy as np
import mxnet as mx
import itertools
from ...util.special import log_determinant
from ...common.config import get_default_MXNet_mode
from ..variables import Variable
from .distribution import Distribution
from .univariate import UnivariateDistribution
from ...runtime.distributions import NormalRuntime
from ...util.inference import broadcast_samples_dict


class Normal(UnivariateDistribution):
    """
    The one-dimensional normal distribution. The normal distribution can be defined over a scalar random variable or an
    array of random variables. In case of an array of random variables, the mean and variance are broadcasted to the
    shape of the output random variable (array).

    :param mean: Mean of the normal distribution.
    :type mean: Variable
    :param variance: Variance of the normal distribution.
    :type variance: Variable
    """
    runtime_dist_class = NormalRuntime

    def __init__(self, mean, variance):
        inputs = [('mean', mean), ('variance', variance)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(Normal, self).__init__(inputs=inputs, outputs=None,
                                     input_names=input_names,
                                     output_names=output_names)

    @staticmethod
    def define_variable(mean, variance, shape=None):
        """
        Creates and returns a random variable drawn from a normal distribution.

        :param mean: Mean of the distribution.
        :param variance: Variance of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the normal distribution.
        :rtypes: Variable
        """
        if shape is None:
            shape = mean.shape
        normal = Normal(mean=mean, variance=variance)
        normal._generate_outputs(shape=shape)
        return normal.random_variable


class NormalMeanPrecision(UnivariateDistribution):
    """
    The one-dimensional normal distribution, parameterized by mean and precision rather than mean and variance.
    The normal distribution can be defined over a scalar random variable
    or an array of random variables. In case of an array of random variables, the mean and precisions are broadcasted
    to the shape of the output random variable (array).

    :param mean: Mean of the normal distribution.
    :type mean: Variable
    :param precision: Precision of the normal distribution.
    :type precision: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    runtime_dist_class = NormalRuntime

    def __init__(self, mean, precision):
        inputs = [('mean', mean), ('precision', precision)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(NormalMeanPrecision, self).__init__(inputs=inputs, outputs=None,
                                                  input_names=input_names,
                                                  output_names=output_names)

    def get_runtime_distribution(self, variables):
        kwargs = self.fetch_runtime_inputs(variables)
        kwargs = broadcast_samples_dict(mx.nd, kwargs)
        mean = kwargs['mean']
        variance = 1/kwargs['precision']
        return self.runtime_dist_class(mean=mean, variance=variance)

    @staticmethod
    def define_variable(mean, precision, shape=None):
        """
        Creates and returns a random variable drawn from a normal distribution.

        :param mean: Mean of the distribution.
        :param precision: Precision of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the normal distribution.
        :rtypes: Variable
        """
        if shape is None:
            shape = mean.shape
        normal = NormalMeanPrecision(mean=mean, precision=precision)
        normal._generate_outputs(shape=shape)
        return normal.random_variable
