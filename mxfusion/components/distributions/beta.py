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


from ...common.config import get_default_MXNet_mode
from .univariate import UnivariateDistribution
from ...runtime.distributions import BetaRuntime


class Beta(UnivariateDistribution):
    """
    The one-dimensional beta distribution. The beta distribution can be defined over a scalar random variable or an
    array of random variables. In case of an array of random variables, a and b are broadcasted to the
    shape of the output random variable (array).

    :param alpha: a parameter (alpha) of the beta distribution.
    :type alpha: Variable
    :param beta: b parameter (beta) of the beta distribution.
    :type beta: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    runtime_dist_class = BetaRuntime

    def __init__(self, alpha, beta, rand_gen=None, dtype=None, ctx=None):
        inputs = [('alpha', alpha), ('beta', beta)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(Beta, self).__init__(inputs=inputs, outputs=None,
                                   input_names=input_names,
                                   output_names=output_names,
                                   rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    @staticmethod
    def define_variable(alpha=1., beta=1., shape=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a beta distribution.

        :param a: The a parameter (alpha) of the distribution.
        :param b: The b parameter (beta) of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the beta distribution.
        :rtypes: Variable
        """
        beta = Beta(alpha=alpha, beta=beta, rand_gen=rand_gen, dtype=dtype,
                    ctx=ctx)
        beta._generate_outputs(shape=shape)
        return beta.random_variable
