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

from mxfusion.common.exceptions import InferenceError
from ...util import special as sp
from ...common.config import get_default_MXNet_mode
from ..variables import Variable
from .distribution import Distribution, LogPDFDecorator, DrawSamplesDecorator
from ..variables import array_has_samples, get_num_samples
from ...util.customop import broadcast_to_w_samples


class WishartLogPDFDecorator(LogPDFDecorator):

    def _wrap_log_pdf_with_broadcast(self, func):
        def log_pdf_broadcast(self, F, **kw):
            """
            Computes the logarithm of the probability density/mass function (PDF/PMF) of the distribution.
            The inputs and outputs variables are in RTVariable format.

            Shape assumptions:
            * degrees_of_freedom is S x 1
            * scale is S x N x D x D
            * random_variable is S x N x D x D

            Where:
            * S, the number of samples, is optional. If more than one of the variables has samples,
            the number of samples in each variable must be the same. S is 1 by default if not a sampled variable.
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

            num_samples = max([get_num_samples(F, v) for v in variables.values()])

            shapes_map = dict(
                degrees_of_freedom=(num_samples,),
                scale=(num_samples,) + rv_shape,
                random_variable=(num_samples,) + rv_shape
            )
            variables = {name: broadcast_to_w_samples(F, v, shapes_map[name])
                         for name, v in variables.items()}
            res = func(self, F=F, **variables)
            return res
        return log_pdf_broadcast


class WishartDrawSamplesDecorator(DrawSamplesDecorator):

    def _wrap_draw_samples_with_broadcast(self, func):
        def draw_samples_broadcast(self, F, rv_shape, num_samples=1,
                                   always_return_tuple=False, **kw):
            """
            Draw a number of samples from the distribution. The inputs and outputs variables are in RTVariable format.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param rv_shape: the shape of each sample
            :type rv_shape: tuple
            :param num_samples: the number of drawn samples (default: one)
            :int num_samples: int
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
                num_samples_inferred = max([get_num_samples(F, v) for v in
                                           variables.values()])
                if num_samples_inferred != num_samples:
                    raise InferenceError("The number of samples in the num_samples argument of draw_samples of "
                                         "the Wishart distribution must be the same as the number of samples "
                                         "given to the inputs. num_samples: {}, the inferred number of samples "
                                         "from inputs: {}.".format(num_samples, num_samples_inferred))

            shapes_map = dict(
                degrees_of_freedom=(num_samples,),
                scale=(num_samples,) + rv_shape,
                random_variable=(num_samples,) + rv_shape)
            variables = {name: broadcast_to_w_samples(F, v, shapes_map[name])
                         for name, v in variables.items()}

            res = func(self, F=F, rv_shape=rv_shape, num_samples=num_samples,
                       **variables)
            if always_return_tuple:
                res = (res,)
            return res
        return draw_samples_broadcast


# noinspection PyPep8Naming
class Wishart(Distribution):
    """
    The Wishart distribution.

    :param degrees_of_freedom: Degrees of freedom of the Wishart distribution.
    :type degrees_of_freedom: Variable
    :param scale: Scale matrix of the distribution.
    :type scale: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, degrees_of_freedom, scale, rand_gen=None, minibatch_ratio=1.,
                 dtype=None, ctx=None):
        inputs = [('degrees_of_freedom', degrees_of_freedom), ('scale', scale)]
        input_names = ['degrees_of_freedom', 'scale']
        output_names = ['random_variable']
        super(Wishart, self).__init__(inputs=inputs, outputs=None,
                                      input_names=input_names,
                                      output_names=output_names,
                                      rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    def replicate_self(self, attribute_map=None):
        """
        Replicates this Factor, using new inputs, outputs, and a new uuid.
        Used during model replication to functionally replicate a factor into a new graph.

        :param inputs: new input variables of the factor.
        :type inputs: a dict of {'name' : Variable} or None
        :param outputs: new output variables of the factor.
        :type outputs: a dict of {'name' : Variable} or None
        """
        replicant = super(Wishart, self).replicate_self(attribute_map)
        return replicant

    @WishartLogPDFDecorator()
    def log_pdf(self, degrees_of_freedom, scale, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (PDF) of the Wishart distribution.

        :param degrees_of_freedom: the degrees of freedom of the Wishart distribution.
        :type degrees_of_freedom: MXNet NDArray or MXNet Symbol
        :param scale: the scale of the distribution.
        :type scale: MXNet NDArray or MXNet Symbol
        :param random_variable: the random variable of the Wishart distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        # Constants
        num_samples, num_data_points, dimension, _ = scale.shape

        # Note that the degrees of freedom should be a float for most of the remaining calculations
        df = degrees_of_freedom.astype(self.dtype)
        a = df - dimension - 1
        b = df * dimension * np.log(2)

        # Make copies of the constants
        df = df.tile(num_data_points).reshape((num_samples, num_data_points))
        a = a.tile(num_data_points).reshape((num_samples, num_data_points))
        b = b.tile(num_data_points).reshape((num_samples, num_data_points))

        log_det_X = sp.log_determinant(random_variable, F)
        log_det_V = sp.log_determinant(scale, F)
        log_gamma_np = sp.log_multivariate_gamma(df / 2, dimension, F)
        tr_v_inv_x = sp.trace(sp.solve(scale, random_variable), F)

        return (0.5 * ((a * log_det_X) - tr_v_inv_x - b - (df * log_det_V)) - log_gamma_np) * self.log_pdf_scaling

    @WishartDrawSamplesDecorator()
    def draw_samples(self, degrees_of_freedom, scale, rv_shape, num_samples=1, F=None):
        """
        Draw a number of samples from the Wishart distribution.

        The Bartlett decomposition of a matrix X from a p-variate Wishart distribution with scale matrix V
        and n degrees of freedom is the factorization:
        X = LAA'L'
        where L is the Cholesky factor of V and A is lower triangular, with
        diagonal elements drawn from a chi squared (n-i+1) distribution where i is the (1-based) diagonal index
        and the off-diagonal elements are independent N(0, 1)

        :param degrees_of_freedom: the degrees of freedom of the Wishart distribution.
        :type degrees_of_freedom: MXNet NDArray or MXNet Symbol
        :param scale: the scale of the Wishart distributions.
        :type scale: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :int num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the Wishart distribution
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        # Cholesky of L
        L = F.linalg.potrf(scale)

        full_shape = (num_samples,) + rv_shape

        import itertools

        # Create the lower triangular matrix A
        A = F.zeros(full_shape, dtype=scale.dtype)
        for i in range(num_samples):
            # Need to have the full shape
            for extra_dims in itertools.product(*map(range, rv_shape[:-2])):
                for j in range(rv_shape[-2]):
                    shape = degrees_of_freedom.asnumpy().ravel()[0] - j
                    ix = (i, ) + extra_dims + (j, j)
                    A[ix] = F.sqrt(F.sum(F.power(F.random.normal(shape=shape), 2)))
                    for k in range(j + 1, rv_shape[-1]):
                        ix = (i,) + extra_dims + (k, j)
                        A[ix] = F.random.normal()

        # Broadcast A
        LA = F.linalg.trmm(L, A)
        samples = F.linalg.gemm2(LA, LA, transpose_b=True)

        return samples

    @staticmethod
    def define_variable(shape, degrees_of_freedom=0, scale=None, rand_gen=None,
                        minibatch_ratio=1., dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a Wishart distribution.

        :param degrees_of_freedom: Degrees of freedom of the distribution.
        :param scale: Scale matrix of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the Wishart distribution.
        :rtypes: Variable
        """
        scale = scale if scale is not None else mx.nd.array(np.eye(N=shape[-1]), dtype=dtype, ctx=ctx)
        wishart = Wishart(degrees_of_freedom=degrees_of_freedom, scale=scale,
                          rand_gen=rand_gen,
                          dtype=dtype, ctx=ctx)
        wishart._generate_outputs(shape=shape)
        return wishart.random_variable

    def _generate_outputs(self, shape):
        """
        Set the output variable of the distribution.

        :param shape: the shape of the random distribution.
        :type shape: tuple
        """
        self.outputs = [('random_variable', Variable(value=self, shape=shape))]
