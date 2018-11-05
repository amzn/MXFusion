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
from .distribution import Distribution, LogPDFDecorator, DrawSamplesDecorator
from ..variables import array_has_samples, get_num_samples
from ...util.customop import broadcast_to_w_samples
from ...common.exceptions import InferenceError


class DirichletLogPDFDecorator(LogPDFDecorator):

    def _wrap_log_pdf_with_broadcast(self, func):
        def log_pdf_broadcast(self, F, **kw):
            """
            Computes the logarithm of the probability density/mass function (PDF/PMF) of the distribution. The inputs
            and outputs variables are in RTVariable format.

            Shape assumptions:
            * a is S x N x D
            * random_variable is S x N x D

            Where:
            * S, the number of samples, is optional. If more than one of the variables has samples, the number of
                samples in each variable must be the same. S is 1 by default if not a sampled variable.
            * N is the number of data points. N can be any number of dimensions (N_1, N_2, ...) but must be
                broadcastable to the shape of random_variable.
            * D is the dimension of the distribution.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param kw: the dict of input and output variables of the distribution
            :type kw: {str (name): MXNet NDArray or MXNet Symbol}
            :returns: log pdf of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol
            """
            variables = {name: kw[name] for name, _ in self.inputs}
            variables['random_variable'] = kw['random_variable']
            rv_shape = variables['random_variable'].shape[1:]

            nSamples = max([get_num_samples(F, v) for v in variables.values()])

            shapes_map = {}
            shapes_map['a'] = (nSamples,) + rv_shape
            shapes_map['random_variable'] = (nSamples,) + rv_shape
            variables = {name: broadcast_to_w_samples(F, v, shapes_map[name])
                         for name, v in variables.items()}
            res = func(self, F=F, **variables)
            return res
        return log_pdf_broadcast


class DirichletDrawSamplesDecorator(DrawSamplesDecorator):

    def _wrap_draw_samples_with_broadcast(self, func):
        def draw_samples_broadcast(self, F, rv_shape, num_samples=1,
                                   always_return_tuple=False, **kw):
            """
            Draw a number of samples from the distribution. The inputs and outputs variables are in RTVariable format.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param tuple rv_shape: the shape of each sample
            :param int num_samples: the number of drawn samples (default: one)
            :param boolean always_return_tuple: Whether return a tuple even if there is only one variables in outputs.
            :param kw: the dict of input variables of the distribution
            :type kw: {name: MXNet NDArray or MXNet Symbol}
            :returns: a set samples of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol or [MXNet NDArray or MXNet Symbol]
            """
            rv_shape = list(rv_shape.values())[0]
            variables = {name: kw[name] for name, _ in self.inputs}

            isSamples = any([array_has_samples(F, v) for v in variables.values()])
            if isSamples:
                num_samples_inferred = max([get_num_samples(F, v) for v in
                                           variables.values()])
                if num_samples_inferred != num_samples:
                    raise InferenceError("The number of samples in the nSamples argument of draw_samples of Dirichlet",
                                         "distribution must be the same as the number of samples given to the inputs. ",
                                         "nSamples: "+str(num_samples)+" the inferred number of samples from inputs: " +
                                         str(num_samples_inferred)+".")

            shapes_map = {}
            shapes_map['a'] = (num_samples,) + rv_shape
            variables = {name: broadcast_to_w_samples(F, v, shapes_map[name])
                         for name, v in variables.items()}

            res = func(self, F=F, rv_shape=rv_shape, num_samples=num_samples,
                       **variables)
            if always_return_tuple:
                res = (res,)
            return res
        return draw_samples_broadcast


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
    def __init__(self, a, normalization=True,
                 rand_gen=None, dtype=None, ctx=None):
        inputs = [('a', a)]
        input_names = ['a']
        output_names = ['random_variable']
        super().__init__(inputs=inputs, outputs=None, input_names=input_names,
                         output_names=output_names, rand_gen=rand_gen,
                         dtype=dtype, ctx=ctx)
        self.normalization = normalization

    @DirichletLogPDFDecorator()
    def log_pdf(self, a, random_variable, F=None):
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
            random_variable = F.broadcast_div(random_variable, F.expand_dims(F.norm(random_variable, ord=1, axis=2),
                                                                             axis=2))
        power = F.broadcast_power(random_variable, a - 1)
        prod = F.prod(power, axis=2)
        beta = F.prod(F.gamma(a), axis=2)/F.gamma(F.sum(a, axis=2))
        logL = F.log(prod/beta)
        return logL

    @DirichletDrawSamplesDecorator()
    def draw_samples(self, a, rv_shape, num_samples=1, F=None):
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

        ones = F.ones_like(a)
        y = self._rand_gen.sample_gamma(alpha=a, beta=ones,
                                        dtype=self.dtype, ctx=self.ctx)
        return F.broadcast_div(y, F.sum(y))

    @staticmethod
    def define_variable(a, shape=None, normalization=True,
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
        dirichlet = Dirichlet(a=a, normalization=normalization,
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
