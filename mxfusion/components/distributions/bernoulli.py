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


from .univariate import UnivariateDistribution
from ...common.config import get_default_MXNet_mode


class Bernoulli(UnivariateDistribution):
    """
    The Bernoulli distribution.

    :param prob_true: the probability of being true.
    :type prob_true: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, prob_true, rand_gen=None, dtype=None, ctx=None):
        inputs = [('prob_true', prob_true)]
        input_names = ['prob_true']
        output_names = ['random_variable']
        super(Bernoulli, self).__init__(
            inputs=inputs, outputs=None,
            input_names=input_names,
            output_names=output_names,
            rand_gen=rand_gen, dtype=dtype,
            ctx=ctx)

    def replicate_self(self, attribute_map=None):
        """
        This functions as a copy constructor for the object.
        In order to do a copy constructor we first call ``__new__`` on the class which creates a blank object.
        We then initialize that object using the methods standard init procedures, and do any extra copying of attributes.

        Replicates this Factor, using new inputs, outputs, and a new uuid.
        Used during model replication to functionally replicate a factor into a new graph.

        :param inputs: new input variables of the factor.
        :type inputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
        :param outputs: new output variables of the factor.
        :type outputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
        """
        replicant = super(Bernoulli, self).replicate_self(attribute_map=attribute_map)
        return replicant

    def log_pdf_impl(self, prob_true, random_variable, F=None):
        """
        Computes the logarithm of probabilistic mass function of the Bernoulli distribution.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param prob_true: the probability of being true.
        :type prob_true: MXNet NDArray or MXNet Symbol
        :param random_variable: the point to compute the logpdf for.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        logL = random_variable * F.log(prob_true) + (1 - random_variable) * F.log(1 - prob_true)
        logL = logL * self.log_pdf_scaling
        return logL

    def draw_samples_impl(self, prob_true, rv_shape, num_samples=1, F=None):
        """
        Draw a number of samples from the Bernoulli distribution.

        :param prob_true: the probability being true.
        :type prob_true: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :int num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the Bernoulli distribution
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        return self._rand_gen.sample_bernoulli(
            prob_true, shape=(num_samples,) + rv_shape, dtype=self.dtype, F=F)

    @staticmethod
    def define_variable(prob_true, shape=None, rand_gen=None, dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a Bernoulli distribution.

        :param prob_true: the probability being true.
        :type prob_true: Variable
        :param shape: the shape of the Bernoulli variable.
        :type shape: tuple of int
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: RandomVariable drawn from the Bernoulli distribution.
        :rtypes: Variable
        """
        bernoulli = Bernoulli(prob_true=prob_true, rand_gen=rand_gen,
                              dtype=dtype, ctx=ctx)
        bernoulli._generate_outputs(shape=shape)
        return bernoulli.random_variable
