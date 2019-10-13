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


from .distribution import Distribution
from ...runtime.distributions import DirichletRuntime


class Dirichlet(Distribution):
    """
    The Dirichlet distribution.

    :param Variable a: alpha, the concentration parameters of the distribution.
    :param boolean normalization: If true, L1 normalization is applied.
    :param RandomGenerator rand_gen: the random generator (default: MXNetRandomGenerator).
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    runtime_dist_class = DirichletRuntime

    def __init__(self, alpha):
        inputs = [('alpha', alpha)]
        input_names = ['alpha']
        output_names = ['random_variable']
        super().__init__(inputs=inputs, outputs=None, input_names=input_names,
                         output_names=output_names)

    @staticmethod
    def define_variable(alpha, shape=None):
        """
        Creates and returns a random variable drawn from a Dirichlet distribution.

        :param Variable a: alpha, the concentration parameters of the distribution.
        :param boolean normalization: If true, L1 normalization is applied.
        :param RandomGenerator rand_gen: the random generator (default: MXNetRandomGenerator).
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the Dirichlet distribution.
        :rtypes: Variable
        """
        dirichlet = Dirichlet(alpha=alpha)
        dirichlet._generate_outputs(shape=shape)
        return dirichlet.random_variable
