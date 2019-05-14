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


from ..variables import Variable
from .distribution import Distribution


class UnivariateDistribution(Distribution):
    """
    The base class of a univariate probability distribution. The univariate
    distribution can be defined over a scalar random variable or an array
    of random variables. In case of an array of random variables, the input
    variables are broadcasted to the shape of the output random variable (array).

    :param inputs: the input variables that parameterize the probability
    distribution.
    :type inputs: {name: Variable}
    :param outputs: the random variables drawn from the distribution.
    :type outputs: {name: Variable}
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, inputs, input_names, output_names, outputs=None,
                 rand_gen=None, dtype=None, ctx=None):
        super(UnivariateDistribution, self).__init__(
            inputs=inputs, outputs=outputs,
            input_names=input_names, output_names=output_names,
            rand_gen=rand_gen, dtype=dtype,
            ctx=ctx)

    def _generate_outputs(self, shape=None):
        """
        Set the output variable of the distribution.

        :param shape: the shape of the random distribution.
        :type shape: tuple
        """
        self.outputs = [('random_variable', Variable(value=self, shape=(1,) if
                        shape is None else shape))]
