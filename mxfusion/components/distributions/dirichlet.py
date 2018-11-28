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
from ...common.config import get_default_MXNet_mode
from .distribution import Distribution


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
    def __init__(self, alpha, normalization=True,
                 rand_gen=None, dtype=None, ctx=None):
        inputs = [('alpha', alpha)]
        input_names = ['alpha']
        output_names = ['random_variable']
        super().__init__(inputs=inputs, outputs=None, input_names=input_names,
                         output_names=output_names, rand_gen=rand_gen,
                         dtype=dtype, ctx=ctx)
        self.normalization = normalization

    def log_pdf_impl(self, alpha, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (pdf) of the Dirichlet distribution.

        :param a: the a parameter (alpha) of the Dirichlet distribution.
        :type a: MXNet NDArray or MXNet Symbol
        :param random_variable: the random variable of the Dirichlet distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        if self.normalization:
            random_variable = F.broadcast_div(
                random_variable, F.expand_dims(F.norm(random_variable, ord=1,
                                               axis=2), axis=2))
        power = F.broadcast_power(random_variable, alpha - 1)
        prod = F.prod(power, axis=2)
        beta = F.prod(F.gamma(alpha), axis=2)/F.gamma(F.sum(alpha, axis=2))
        logL = F.log(prod/beta)
        return logL

    def draw_samples_impl(self, alpha, rv_shape, num_samples=1, F=None):
        """
        Draw samples from the Dirichlet distribution.

        :param a: the a parameter (alpha) of the Dirichlet distribution.
        :type a: MXNet NDArray or MXNet Symbol
        :param tuple rv_shape: the shape of each sample (this variable is not used because the shape of the random var
            is given by the shape of a)
        :param int num_samples: the number of drawn samples (default: one).
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the Dirichlet distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        ones = F.ones_like(alpha)
        y = self._rand_gen.sample_gamma(alpha=alpha, beta=ones,
                                        dtype=self.dtype, ctx=self.ctx)
        return F.broadcast_div(y, F.sum(y))

    @staticmethod
    def define_variable(alpha, shape=None, normalization=True,
                        rand_gen=None, dtype=None, ctx=None):
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
        dirichlet = Dirichlet(alpha=alpha, normalization=normalization,
                              rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        dirichlet._generate_outputs(shape=shape)
        return dirichlet.random_variable

    def _generate_outputs(self, shape):
        """
        Set the output variable of the distribution.

        :param shape: the shape of the random distribution.
        :type shape: tuple
        """
        self.outputs = [('random_variable', Variable(value=self, shape=shape))]

    def replicate_self(self, attribute_map=None):
        """
        This functions as a copy constructor for the object.
        In order to do a copy constructor we first call ``__new__`` on the class which creates a blank object.
        We then initialize that object using the methods standard init procedures, and do any extra copying of
        attributes.

        Replicates this Factor, using new inputs, outputs, and a new uuid.
        Used during model replication to functionally replicate a factor into a new graph.

        :param inputs: new input variables of the factor.
        :type inputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
        :param outputs: new output variables of the factor.
        :type outputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
        """
        replicant = super().replicate_self(attribute_map=attribute_map)
        replicant.normalization = self.normalization
        return replicant
