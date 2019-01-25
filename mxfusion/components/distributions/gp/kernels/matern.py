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
from .stationary import StationaryKernel


class Matern(StationaryKernel):
    """
    Matern kernel:

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

    def __init__(self, input_dim, order, ARD=False, variance=1.,
                 lengthscale=1., name='matern', active_dims=None, dtype=None,
                 ctx=None):
        super(Matern, self).__init__(
            input_dim=input_dim, ARD=ARD, variance=variance,
            lengthscale=lengthscale, name=name, active_dims=active_dims,
            dtype=dtype, ctx=ctx)
        self.order = order


class Matern52(Matern):
    def __init__(self, input_dim, ARD=False, variance=1., lengthscale=1.,
                 name='matern52', active_dims=None, dtype=None, ctx=None):
        super(Matern52, self).__init__(
            input_dim=input_dim, order=2, ARD=ARD, variance=variance,
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
        R = F.sqrt(F.clip(R2, 1e-14, np.inf))
        return F.broadcast_mul(
            (1+np.sqrt(5)*R+5/3.*R2)*F.exp(-np.sqrt(5)*R),
            F.expand_dims(variance, axis=-2))


class Matern32(Matern):
    def __init__(self, input_dim, ARD=False, variance=1., lengthscale=1.,
                 name='matern32', active_dims=None, dtype=None, ctx=None):
        super(Matern32, self).__init__(
            input_dim=input_dim, order=1, ARD=ARD, variance=variance,
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
        R = F.sqrt(F.clip(R2, 1e-14, np.inf))
        return F.broadcast_mul(
            (1+np.sqrt(3)*R)*F.exp(-np.sqrt(3)*R),
            F.expand_dims(variance, axis=-2))


class Matern12(Matern):
    def __init__(self, input_dim, ARD=False, variance=1., lengthscale=1.,
                 name='matern12', active_dims=None, dtype=None, ctx=None):
        super(Matern12, self).__init__(
            input_dim=input_dim, order=0, ARD=ARD, variance=variance,
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
        R = F.sqrt(F.clip(self._compute_R2(F, X, lengthscale, variance, X2=X2),
                          1e-14, np.inf))
        return F.broadcast_mul(
            F.exp(-R), F.expand_dims(variance, axis=-2))
