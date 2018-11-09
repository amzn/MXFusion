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


from copy import copy
from .....common.exceptions import ModelSpecificationError
from .....util.util import rename_duplicate_names, slice_axis
from ....variables import Variable
from ....functions.mxfusion_function import MXFusionFunction

# TODO: write the design doc for kernels


class Kernel(MXFusionFunction):
    """
    The base class for a Gaussian process kernel: a positive definite function which forms of a covariance function (kernel).

    :param input_dim: the number of dimensions of the kernel. (The total number of active dimensions).
    :type input_dim: int
    :param name: the name of the kernel. The name is also used as the prefix for the kernel parameters.
    :type name: str
    :param active_dims: The dimensions of the inputs that are taken for the covariance matrix computation. (default: None, taking all the dimensions).
    :type active_dims: [int] or None
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    broadcastable = False

    def __init__(self, input_dim, name, active_dims=None, dtype=None,
                 ctx=None):
        super(Kernel, self).__init__(
            func_name=name, dtype=dtype, broadcastable=self.broadcastable)
        self.input_dim = input_dim
        self.ctx = ctx
        self.active_dims = active_dims
        self._parameter_names = []

    def __setattr__(self, name, value):
        """
        Override to maintain a list of kernel parameters.
        """
        if isinstance(value, Variable):
            if name not in self._parameter_names:
                self._parameter_names.append(name)
        super(Kernel, self).__setattr__(name, value)

    @property
    def local_parameters(self):
        """
        The kernel parameters in the current kernel, which does not include kernel parameters that belongs to the sub-kernels of a compositional
        kernel. The keys of the returned dictionary are the name of the kernel parameters (without the prefix) and the values are the corresponding
        variables.

        :return: a dictionary of local kernel parameters, in which the keys are the name of individual parameters, including the kernel in front, and
            the values are the corresponding Variables.
        :rtype: {str: Variable}
        """
        return {getattr(self, n) for n in self._parameter_names}

    @property
    def parameters(self):
        """
        All the kernel parameters including the kernel parameters that belongs to the sub-kernels. The keys of the returned dictionary are the name of
        the kernel parameters with a prefix (the name of the kernel plus '_') and the values are the corresponding variables.

        :return: a dictionary of all the kernel parameters, in which the keys are the name of individual parameters, including the kernel in front,
            and the values are the corresponding Variables.
        :rtype: {str: Variable}
        """
        raise NotImplementedError

    @property
    def input_names(self):
        return ['X', 'X2']

    @property
    def output_names(self):
        return ['covariance']

    def K(self, F, X, X2=None, **kernel_params):
        """
        Compute the covariance matrix.

        .. math::
            K_{ij} = k(X_i, X2_j)

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param X2: (optional) the second set of arguments to the kernel. If X2 is None, this computes a square covariance matrix of X. In other words,
            X2 is internally treated as X.
        :type X2: MXNet NDArray or MXNet Symbol
        :param **kernel_params: the set of kernel parameters, provided as keyword arguments.
        :type **kernel_params: {str: MXNet NDArray or MXNet Symbol}
        :return: The covariance matrix
        :rtype: MXNet NDArray or MXNet Symbol
        """
        # Remove the prefix in the names of kernel_params
        # The prefix is defined as the kernel name plus _
        offset = len(self.name) + 1
        params = {k[offset:]: v for k, v in kernel_params.items() if
                  k.startswith(self.name + '_')}
        if self.active_dims is not None:
            X = slice_axis(F, X, axis=-1, indices=self.active_dims)
            if X2 is not None:
                X2 = slice_axis(F, X2, axis=-1, indices=self.active_dims)
        return self._compute_K(F=F, X=X, X2=X2, **params)

    def Kdiag(self, F, X, **kernel_params):
        """
        Compute the diagonal of the covariance matrix.

        .. math::
            K_{ii} = k(X_i, X_i)

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param **kernel_params: the set of kernel parameters, provided as keyword arguments.
        :type **kernel_params: {str: MXNet NDArray or MXNet Symbol}
        :return: The diagonal of the covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        # Remove the prefix in the names of kernel_params
        # The prefix is defined as the kernel name plus _
        offset = len(self.name) + 1
        params = {k[offset:]: v for k, v in kernel_params.items() if
                  k.startswith(self.name + '_')}
        if self.active_dims is not None:
            X = slice_axis(F, X, axis=-1, indices=self.active_dims)
        return self._compute_Kdiag(F=F, X=X, **params)

    def add(self, other, name='add'):
        """
        Construct a new kernel by adding this kernel to another kernel.

        :param other: the other kernel to be added.
        :type other: Kernel
        :return: the kernel which is the sum of the current kernel with the specified kernel.
        :rtype: Kernel
        """
        if not isinstance(other, Kernel):
            raise ModelSpecificationError("Only a Gaussian Process Kernel can be added to a Gaussian Process Kernel.")
        from .add_kernel import AddKernel
        return AddKernel([self, other], name=name, ctx=self.ctx,
                         dtype=self.dtype)

    def __add__(self, other):
        """
        Overwrite the "+" operator to perform summation of kernels.
        """
        return self.add(other)

    def multiply(self, other, name='mul'):
        """
        Construct a new kernel by multiplying this kernel with another kernel.

        :param other: the other kernel to be added.
        :type other: Kernel
        :return: the kernel which is the sum of the current kernel with the specified kernel.
        :rtype: Kernel
        """
        if not isinstance(other, Kernel):
            raise ModelSpecificationError(
                "Only a Gaussian Process Kernel can be multiplied with a Gaussian Process Kernel.")
        from .multiply_kernel import MultiplyKernel
        return MultiplyKernel([self, other], name=name, ctx=self.ctx,
                         dtype=self.dtype)

    def __mul__(self, other):
        """
        Overload the "*" operator to perform multiplication of kernels
        """
        return self.multiply(other)

    def _compute_K(self, F, X, X2=None, **kernel_params):
        """
        The internal interface for the actual covariance matrix computation.

        This function takes as an assumption: The prefix in the keys of *kernel_params* that corresponds to the name of the kernel has been
        removed. The dimensions of *X* and *X2* have been sliced according to *active_dims*.

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
        raise NotImplementedError

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
        raise NotImplementedError

    def fetch_parameters(self, params):
        """
        The helper function to fetch the kernel parameters from a set of variables according to the UUIDs of the kernel parameters. It returns a
        dictionary of kernel parameters, where the keys are the name of the kernel parameters and the values are the MXNet array at runtime. The
        returned dict can be directly passed into *K* and *Kdiag* as *kernel_params*.

        :param params: the set of parameters where the kernel parameters are fetched from.
        :type params: {str (UUID): MXNet NDArray or MXNet Symbol}
        :return: a dict of the kernel parameters, where the keys are the name of the kernel parameters and the values are the MXNet array at runtime.
        :rtype: {str (kernel name): MXNet NDArray or MXNet Symbol}
        """
        return {n: params[v.uuid] for n, v in self.parameters.items()}

    def eval(self, F, X, X2=None, **kernel_params):
        """
        The method handling the execution of the function.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
        :param **input_kws: the dict of inputs to the functions. The key in the
        dict should match with the name of inputs specified in the inputs of
        FunctionEvaluation.
        :type **input_kws: {variable name: MXNet NDArray or MXNet Symbol}
        :returns: the return value of the function
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        return self.K(F, X, X2, **kernel_params)

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for a kernel.
        """
        replicant = super(Kernel, self).replicate_self(attribute_map)
        replicant.input_dim = self.input_dim
        replicant.ctx = self.ctx
        replicant.active_dims = copy(self.active_dims)
        replicant._parameter_names = []
        for n in self._parameter_names:
            setattr(replicant, n, getattr(
                self, n).replicate_self(attribute_map))
        return replicant


class NativeKernel(Kernel):
    """
    The base class for all the native kernels: the computation of the covariance matrix does not depend on other kernels.

    :param input_dim: the number of dimensions of the kernel. (The total number of active dimensions).
    :type input_dim: int
    :param name: the name of the kernel. The name is used to access kernel parameters.
    :type name: str
    :param active_dims: The dimensions of the inputs that are taken for the covariance matrix computation. (default: None, taking all the dimensions).
    :type active_dims: [int] or None
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, input_dim, name, active_dims=None, dtype=None,
                 ctx=None):
        super(NativeKernel, self).__init__(input_dim=input_dim, name=name,
                                           active_dims=active_dims,
                                           dtype=dtype, ctx=ctx)

    @property
    def parameters(self):
        """
        All the kernel parameters including the kernel parameters that belongs to the sub-kernels. The keys of the returned dictionary are the name of
        the kernel parameters with a prefix (the name of the kernel plus '_') and the values are the corresponding variables.

        :return: a dictionary of all the kernel parameters, in which the keys are the name of individual parameters, including the kernel in front,
            and the values are the corresponding Variables.
        :rtype: {str: Variable}
        """
        return {self.name + '_' + n: getattr(self, n) for n in
                self._parameter_names}

    @property
    def parameter_names(self):
        return [self.name + '_' + n for n in self._parameter_names]


class CombinationKernel(Kernel):
    """
    The base class for combination kernels: the covariance matrix is computed by combining the covariance matrix from multiple sub-kernels.

    :param sub_kernels: a list of kernels that are combined to compute a covariance matrix.
    :type sub_kernels: [Kernel]
    :param name: the name of the kernel. The name is used to access kernel parameters.
    :type name: str
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, sub_kernels, name, dtype=None, ctx=None):
        input_dim = max([k.input_dim for k in sub_kernels])
        renaming = rename_duplicate_names([k.name for k in sub_kernels])
        for i, n in renaming:
            sub_kernels[i].name = n
        super(CombinationKernel, self).__init__(
            input_dim=input_dim, name=name, dtype=dtype, ctx=ctx)
        self.sub_kernels = sub_kernels
        for k in self.sub_kernels:
            setattr(self, k.name, k)

    @property
    def parameters(self):
        """
        All the kernel parameters including the kernel parameters that belongs to the sub-kernels. The keys of the returned dictionary are the name of
        the kernel parameters with a prefix (the name of the kernel plus '_') and the values are the corresponding variables.

        :return: a dictionary of all the kernel parameters, in which the keys are the name of individual parameters, including the kernel in front,
            and the values are the corresponding Variables.
        :rtype: {str: Variable}
        """
        p = {}
        for k in self.sub_kernels:
            p.update(k.parameters)
        return {self.name + '_' + k: v for k, v in p.items()}

    @property
    def parameter_names(self):
        pnames = []
        for k in self.sub_kernels:
            pnames.extend([self.name + '_' + k for k in k.parameter_names])
        return pnames

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for a kernel.
        """
        replicant = super(CombinationKernel, self).replicate_self(
            attribute_map)
        replicant.sub_kernels = [k.replicate_self(attribute_map) for k in
                                 self.sub_kernels]
        for k in replicant.sub_kernels:
            setattr(replicant, k.name, k)
        return replicant
