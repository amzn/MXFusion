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
from .univariate import UnivariateDistribution
from .distribution import LogPDFDecorator, DrawSamplesDecorator
from ...util.customop import broadcast_to_w_samples
from ..variables import get_num_samples, array_has_samples
from ...common.config import get_default_MXNet_mode
from ...common.exceptions import InferenceError


class BernoulliLogPDFDecorator(LogPDFDecorator):

    def _wrap_log_pdf_with_broadcast(self, func):
        def log_pdf_broadcast(self, F, **kw):
            """
            Computes the logarithm of the probability density/mass function (PDF/PMF) of the distribution.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param kw: the dict of input and output variables of the distribution
            :type kw: {name: MXNet NDArray or MXNet Symbol}
            :returns: log pdf of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol
            """
            variables = {name: kw[name] for name, _ in self.inputs}
            variables['random_variable'] = kw['random_variable']
            rv_shape = variables['random_variable'].shape[1:]

            n_samples = max([get_num_samples(F, v) for v in variables.values()])
            full_shape = (n_samples,) + rv_shape

            variables = {
                name: broadcast_to_w_samples(F, v, full_shape[:-1]+(v.shape[-1],)) for name, v in variables.items()}
            res = func(self, F=F, **variables)
            return res
        return log_pdf_broadcast


class BernoulliDrawSamplesDecorator(DrawSamplesDecorator):

    def _wrap_draw_samples_with_broadcast(self, func):
        def draw_samples_broadcast(self, F, rv_shape, num_samples=1,
                                   always_return_tuple=False, **kw):
            """
            Draw a number of samples from the distribution.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param rv_shape: the shape of each sample
            :type rv_shape: tuple
            :param num_samples: the number of drawn samples (default: one)
            :int n_samples: int
            :param always_return_tuple: Whether return a tuple even if there is only one variables in outputs.
            :type always_return_tuple: boolean
            :param kw: the dict of input variables of the distribution
            :type kw: {name: MXNet NDArray or MXNet Symbol}
            :returns: a set samples of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol or [MXNet NDArray or MXNet Symbol]
            """
            rv_shape = list(rv_shape.values())[0]
            variables = {name: kw[name] for name, _ in self.inputs}

            is_samples = any([array_has_samples(F, v) for v in variables.values()])
            if is_samples:
                num_samples_inferred = max([get_num_samples(F, v) for v in variables.values()])
                if num_samples_inferred != num_samples:
                    raise InferenceError("The number of samples in the n_samples argument of draw_samples of "
                                         "Bernoulli has to be the same as the number of samples given "
                                         "to the inputs. n_samples: {} the inferred number of samples from "
                                         "inputs: {}.".format(num_samples, num_samples_inferred))
            full_shape = (num_samples,) + rv_shape

            variables = {
                name: broadcast_to_w_samples(F, v, full_shape[:-1]+(v.shape[-1],)) for name, v in
                variables.items()}
            res = func(self, F=F, rv_shape=rv_shape, num_samples=num_samples,
                       **variables)
            if always_return_tuple:
                res = (res,)
            return res
        return draw_samples_broadcast


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

    @BernoulliLogPDFDecorator()
    def log_pdf(self, prob_true, random_variable, F=None):
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

    @BernoulliDrawSamplesDecorator()
    def draw_samples(self, prob_true, rv_shape, num_samples=1, F=None):
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
        return self._rand_gen.sample_bernoulli(prob_true, shape=(num_samples,) + rv_shape, dtype=self.dtype, F=F)

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
        bernoulli = Bernoulli(prob_true=prob_true, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        bernoulli._generate_outputs(shape=shape)
        return bernoulli.random_variable
