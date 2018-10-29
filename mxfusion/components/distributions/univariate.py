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


from ...common.exceptions import InferenceError
from ..variables import Variable
from .distribution import Distribution, LogPDFDecorator, DrawSamplesDecorator
from ..variables import array_has_samples, get_num_samples
from ...util.customop import broadcast_to_w_samples


class UnivariateLogPDFDecorator(LogPDFDecorator):

    def _wrap_log_pdf_with_broadcast(self, func):
        def log_pdf_broadcast(self, F, **kws):
            """
            Computes the logarithm of the probability density/mass function
            (PDF/PMF) of the distribution.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param kws: the dict of input and output variables of the distribution
            :type kws: {name: MXNet NDArray or MXNet Symbol}
            :returns: log pdf of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol
            """
            variables = {name: kws[name] for name, _ in self.inputs}
            variables['random_variable'] = kws['random_variable']
            rv_shape = variables['random_variable'].shape[1:]

            num_samples = max([get_num_samples(F, v) for v in variables.values()])
            full_shape = (num_samples,) + rv_shape

            variables = {name: broadcast_to_w_samples(F, v, full_shape) for
                         name, v in variables.items()}
            res = func(self, F=F, **variables)
            return res
        return log_pdf_broadcast


class UnivariateDrawSamplesDecorator(DrawSamplesDecorator):

    def _wrap_draw_samples_with_broadcast(self, func):
        def draw_samples_broadcast(self, F, rv_shape, num_samples=1,
                                   always_return_tuple=False, **kws):
            """
            Draw a number of samples from the distribution.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param rv_shape: the shape of each sample
            :type rv_shape: tuple
            :param nSamples: the number of drawn samples (default: one)
            :int nSamples: int
            :param always_return_tuple: Whether return a tuple even if there is only one variables in outputs.
            :type always_return_tuple: boolean
            :param kws: the dict of input variables of the distribution
            :type kws: {name: MXNet NDArray or MXNet Symbol}
            :returns: a set samples of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol or [MXNet NDArray or MXNet Symbol]
            """
            rv_shape = list(rv_shape.values())[0]
            variables = {name: kws[name] for name, _ in self.inputs}

            isSamples = any([array_has_samples(F, v) for v in variables.values()])
            if isSamples:
                num_samples_inferred = max([get_num_samples(F, v) for v in
                                           variables.values()])
                if num_samples_inferred != num_samples:
                    raise InferenceError("The number of samples in the nSamples argument of draw_samples of Gaussian process has to be the same as the number of samples given to the inputs. nSamples: "+str(num_samples)+" the inferred number of samples from inputs: "+str(num_samples_inferred)+".")
            full_shape = (num_samples,) + rv_shape

            variables = {name: broadcast_to_w_samples(F, v, full_shape) for
                         name, v in variables.items()}
            res = func(self, F=F, rv_shape=rv_shape, num_samples=num_samples,
                       **variables)
            if always_return_tuple:
                res = (res,)
            return res
        return draw_samples_broadcast


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
