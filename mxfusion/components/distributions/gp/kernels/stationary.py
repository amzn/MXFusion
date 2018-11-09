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


from .kernel import NativeKernel
from ....variables import Variable
from ....variables import PositiveTransformation
from .....util.customop import broadcast_to_w_samples


class StationaryKernel(NativeKernel):
    """
    The base class for Stationary kernels (covariance functions).

    Stationary kernels (covariance functions).
    Stationary covariance function depend only on r^2, where r^2 is defined as
    .. math::
        r2(x, x') = \\sum_{q=1}^Q (x_q - x'_q)^2
    The covariance function k(x, x' can then be written k(r).

    In this implementation, r is scaled by the lengthscales parameter(s):
    .. math::
        r2(x, x') = \\sum_{q=1}^Q \\frac{(x_q - x'_q)^2}{\\ell_q^2}.
    By default, there's only one lengthscale: separate lengthscales for each dimension can be enables by setting ARD=True.


    :param input_dim: the number of dimensions of the kernel. (The total number of active dimensions).
    :type input_dim: int
    :param ARD: a binary switch for Automatic Relevance Determination (ARD). If true, the squared distance is divided by a lengthscale for individual
        dimensions.
    :type ARD: boolean
    :param variance: the initial value for the variance parameter (scalar), which scales the whole covariance matrix.
    :type variance: float or MXNet NDArray
    :param lengthscale: the initial value for the lengthscale parameter.
    :type lengthscale: float or MXNet NDArray
    :param name: the name of the kernel. The name is used to access kernel parameters.
    :type name: str
    :param active_dims: The dimensions of the inputs that are taken for the covariance matrix computation. (default: None, taking all the dimensions).
    :type active_dims: [int] or None
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, input_dim, ARD=False, variance=1., lengthscale=1.,
                 name='stationary', active_dims=None, dtype=None, ctx=None):
        super(StationaryKernel, self).__init__(
            input_dim=input_dim, name=name, active_dims=active_dims,
            dtype=dtype, ctx=ctx)
        self.ARD = ARD
        if not isinstance(variance, Variable):
            variance = Variable(shape=(1,),
                                transformation=PositiveTransformation(),
                                initial_value=variance)
        if not isinstance(lengthscale, Variable):
            lengthscale = Variable(shape=(input_dim if ARD else 1,),
                                   transformation=PositiveTransformation(),
                                   initial_value=lengthscale)
        self.variance = variance
        self.lengthscale = lengthscale

    def _compute_R2(self, F, X, lengthscale, variance, X2=None):
        """
        The helper function that computes the squared distance for a stationary kernel.

        Compute the covariance matrix.

        .. math::
            r2(x, x') = \\sum_{q=1}^Q \\frac{(x_q - x'_q)^2}{\\ell_q^2}.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param X2: (optional) the second set of arguments to the kernel. If X2 is None, this computes a square covariance matrix of X. In other words,
            X2 is internally treated as X.
        :type X2: MXNet NDArray or MXNet Symbol
        :return: The squared distance.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        lengthscale = F.expand_dims(lengthscale, axis=-2)
        if X2 is None:
            xsc = X / broadcast_to_w_samples(F, lengthscale, X.shape)
            amat = F.linalg.syrk(xsc) * -2
            dg_a = F.sum(F.square(xsc), axis=-1)
            amat = F.broadcast_add(amat, F.expand_dims(dg_a, axis=-1))
            amat = F.broadcast_add(amat, F.expand_dims(dg_a, axis=-2))
        else:
            x1sc = X / broadcast_to_w_samples(F, lengthscale, X.shape)
            x2sc = X2 / broadcast_to_w_samples(F, lengthscale, X2.shape)
            amat = F.linalg.gemm2(x1sc, x2sc, False, True) * -2
            dg1 = F.sum(F.square(x1sc), axis=-1, keepdims=True)
            amat = F.broadcast_add(amat, dg1)
            dg2 = F.expand_dims(F.sum(F.square(x2sc), axis=-1), axis=-2)
            amat = F.broadcast_add(amat, dg2)
        return amat

    def _compute_Kdiag(self, F, X, lengthscale, variance):
        """
        The internal interface for the actual computation for the diagonal of the covariance matrix.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param variance: the variance parameter (scalar), which scales the whole covariance matrix.
        :type variance: MXNet NDArray or MXNet Symbol
        :param lengthscale: the lengthscale parameter.
        :type lengthscale: MXNet NDArray or MXNet Symbol
        :return: The covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        return broadcast_to_w_samples(F, variance, X.shape[:-1])

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for a kernel.
        """
        replicant = super(StationaryKernel, self).replicate_self(attribute_map)
        replicant.ARD = self.ARD
        return replicant
