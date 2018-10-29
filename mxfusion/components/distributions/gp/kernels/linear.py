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


class Linear(NativeKernel):
    """
    Linear kernel

    .. math::
       k(x,y) = \\sum_{i=1}^{\\text{input_dim}} \\sigma^2_i x_iy_i

    :param input_dim: the number of dimensions of the kernel. (The total number of active dimensions) .
    :type input_dim: int
    :param ARD: a binary switch for Automatic Relevance Determination (ARD). If true, the squared distance is divided by a lengthscale for individual
        dimensions.
    :type ARD: boolean
    :param variances: the initial value for the variances parameter, which scales the input dimensions.
    :type variances: float or MXNet NDArray
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

    def __init__(self, input_dim, ARD=False, variances=1., name='linear',
                 active_dims=None, dtype=None, ctx=None):
        super(Linear, self).__init__(input_dim=input_dim, name=name,
                                     active_dims=active_dims, dtype=dtype,
                                     ctx=ctx)
        self.ARD = ARD
        if not isinstance(variances, Variable):
            variances = Variable(shape=(input_dim if ARD else 1,),
                                 transformation=PositiveTransformation(),
                                 initial_value=variances)
        self.variances = variances

    def _compute_K(self, F, X, variances, X2=None):
        """
        The internal interface for the actual covariance matrix computation.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param X2: (optional) the second set of arguments to the kernel. If X2 is None, this computes a square covariance matrix of X. In other words,
            X2 is internally treated as X.
        :type X2: MXNet NDArray or MXNet Symbol
        :param variances: the variances parameter, which scales the input dimensions.
        :type variances: MXNet NDArray or MXNet Symbol
        :param lengthscale: the lengthscale parameter.
        :type lengthscale: MXNet NDArray or MXNet Symbol
        :return: The covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        if self.ARD:
            var_sqrt = F.sqrt(variances)
            if X2 is None:
                xsc = X * broadcast_to_w_samples(F, var_sqrt, X.shape)
                return F.linalg.syrk(xsc)
            else:
                xsc = X * broadcast_to_w_samples(F, var_sqrt, X.shape)
                x2sc = X2 * broadcast_to_w_samples(F, var_sqrt, X2.shape)
                return F.linalg.gemm2(xsc, x2sc, False, True)
        else:
            if X2 is None:
                A = F.linalg.syrk(X)
            else:
                A = F.linalg.gemm2(X, X2, False, True)
            return A * broadcast_to_w_samples(F, variances, A.shape)

    def _compute_Kdiag(self, F, X, variances):
        """
        The internal interface for the actual computation for the diagonal of the covariance matrix.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param variances: the variances parameter, which scales the input dimensions.
        :type variances: MXNet NDArray or MXNet Symbol
        :return: The covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        X2 = F.square(X)
        return F.sum(X2 * broadcast_to_w_samples(F, variances, X2.shape),
                     axis=-1)

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for a kernel.
        """
        replicant = super(Linear, self).replicate_self(attribute_map)
        replicant.ARD = self.ARD
        return replicant
