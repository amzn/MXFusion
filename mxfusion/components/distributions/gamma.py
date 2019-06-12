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


import mxnet as mx
from .univariate import UnivariateDistribution
from ...util.inference import broadcast_samples_dict
from ...runtime.distributions import GammaRuntime


class Gamma(UnivariateDistribution):
    """
    Gamma distribution parameterized using Alpha and Beta.
    Takes dependency on Scipy to compute the log-gamma function.

    :param alpha: the alpha parameter of the Gamma distribution.
    :type alpha: Variable
    :param beta: beta parameter of the Gamma distribution.
    :type beta: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    runtime_dist_class = GammaRuntime

    def __init__(self, alpha, beta):
        inputs = [('alpha', alpha), ('beta', beta)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(Gamma, self).__init__(inputs=inputs, outputs=None,
                                    input_names=input_names,
                                    output_names=output_names)

    @staticmethod
    def define_variable(alpha=0., beta=1., shape=None):
        """
        Creates and returns a random variable drawn from a Gamma distribution parameterized with a and b parameters.

        :param alpha: beta parameter of the Gamma random variable (also known as rate)
        :type alpha: float
        :param beta: alpha parameter of the Gamma random variable (also known as shape)
        :type beta: float
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the Gamma distribution.
        :rtypes: Variable
        """
        dist = Gamma(alpha=alpha, beta=beta)
        dist._generate_outputs(shape=shape)
        return dist.random_variable


class GammaMeanVariance(UnivariateDistribution):
    """
    Gamma distribution parameterized using Mean and Variance.
    Takes dependency on Scipy to compute the log-gamma function.

    :param mean: the mean parameter of the Gamma distribution.
    :type mean: Variable
    :param variance: variance parameter of the Gamma distribution.
    :type variance: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    runtime_dist_class = GammaRuntime

    def __init__(self, mean, variance):
        inputs = [('mean', mean), ('variance', variance)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(GammaMeanVariance, self).__init__(
            inputs=inputs, outputs=None, input_names=input_names,
            output_names=output_names)

    def get_runtime_distribution(self, variables):
        if self.runtime_dist_class is None:
            raise NotImplementedError
        kwargs = self.fetch_runtime_inputs(variables)
        kwargs = broadcast_samples_dict(mx.nd, kwargs)
        alpha = kwargs['mean']**2 / kwargs['variance']
        beta = kwargs['mean'] / kwargs['variance']
        return self.runtime_dist_class(alpha=alpha, beta=beta)

    @staticmethod
    def define_variable(mean=0., variance=1., shape=None):
        """
        Creates and returns a random variable drawn from a Gamma distribution parameterized with mean and variance.

        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the Gamma distribution.
        :rtypes: Variable
        """
        dist = GammaMeanVariance(mean=mean, variance=variance)
        dist._generate_outputs(shape=shape)
        return dist.random_variable
