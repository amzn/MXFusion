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


from .kernel import CombinationKernel


class AddKernel(CombinationKernel):
    """
    The add kernel that computes a covariance matrix by summing the covariance
    matrices of a list of kernels.

    :param sub_kernels: a list of kernels that are combined to compute a covariance matrix.
    :type sub_kernels: [Kernel]
    :param name: the name of the kernel. The name is used to access kernel parameters.
    :type name: str
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, sub_kernels, name='add', dtype=None, ctx=None):
        kernels = []
        for k in sub_kernels:
            if isinstance(k, AddKernel):
                for k2 in k.sub_kernels:
                    kernels.append(k2)
            else:
                kernels.append(k)
        super(AddKernel, self).__init__(
            sub_kernels=kernels, name=name, dtype=dtype, ctx=ctx)

    def _compute_K(self, F, X, X2=None, **kernel_params):
        """
        The internal interface for the actual covariance matrix computation.

        This function takes as an assumption: The prefix in the keys of
        *kernel_params* that corresponds to the name of the kernel has been
        removed. The dimensions of *X* and *X2* have been sliced according to
        *active_dims*.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param X2: (optional) the second set of arguments to the kernel. If X2 is None, this computes a square covariance matrix of X. In other words,
            X2 is internally treated as X.
        :type X2: MXNet NDArray or MXNet Symbol
        :param **kernel_params: the set of kernel parameters, provided as keyword arguments.
        :type **kernel_params: {str: MXNet NDArray or MXNet Symbol}
        :return: The covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        K = self.sub_kernels[0].K(F=F, X=X, X2=X2, **kernel_params)
        for k in self.sub_kernels[1:]:
            K = K + k.K(F=F, X=X, X2=X2, **kernel_params)
        return K

    def _compute_Kdiag(self, F, X, **kernel_params):
        """
        The internal interface for the actual computation for the diagonal of the covariance matrix.

        This function takes as an assumption: The prefix in the keys of *kernel_params* that corresponds to the name of the kernel has been
        removed. The dimensions of *X* has been sliced according to *active_dims*.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param **kernel_params: the set of kernel parameters, provided as keyword arguments.
        :type **kernel_params: {str: MXNet NDArray or MXNet Symbol}
        :return: The covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        K = self.sub_kernels[0].Kdiag(F=F, X=X, **kernel_params)
        for k in self.sub_kernels[1:]:
            K = K + k.Kdiag(F=F, X=X, **kernel_params)
        return K
