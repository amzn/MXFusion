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


from .stationary import StationaryKernel
from .....util.customop import broadcast_to_w_samples


class RBF(StationaryKernel):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::
       k(r^2) = \\sigma^2 \\exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

    :param input_dim: the number of dimensions of the kernel. (The total number of active dimensions)
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
    broadcastable = True

    def __init__(self, input_dim, ARD=False, variance=1., lengthscale=1.,
                 name='rbf', active_dims=None, dtype=None, ctx=None):
        super(RBF, self).__init__(
            input_dim=input_dim, ARD=ARD, variance=variance,
            lengthscale=lengthscale, name=name, active_dims=active_dims,
            dtype=dtype, ctx=ctx)

    def _compute_K(self, F, X, lengthscale, variance, X2=None):
        """
        The internal interface for the actual covariance matrix computation.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param X2: (optional) the second set of arguments to the kernel. If X2 is None, this computes a square covariance matrix of X. In other words,
            X2 is internally treated as X.
        :type X2: MXNet NDArray or MXNet Symbol
        :param variance: the variance parameter (scalar), which scales the whole covariance matrix.
        :type variance: MXNet NDArray or MXNet Symbol
        :param lengthscale: the lengthscale parameter.
        :type lengthscale: MXNet NDArray or MXNet Symbol
        :return: The covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        R2 = self._compute_R2(F, X, lengthscale, variance, X2=X2)
        return F.exp(R2 / -2) * broadcast_to_w_samples(F, variance, R2.shape)
