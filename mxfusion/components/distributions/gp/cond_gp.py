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
from ....common.config import get_default_MXNet_mode
from ....common.exceptions import ModelSpecificationError
from ...variables.variable import Variable
from ..distribution import Distribution


class ConditionalGaussianProcess(Distribution):
    """
    The Conditional Gaussian process distribution.

    A Gaussian process consists of a kernel function and a mean function (optional). A collection of GP random variables follows a multi-variate
    normal distribution, where the mean is computed from the mean function (zero, if not given) and the covariance matrix is computed from the kernel
    function, both of which are computed given a collection of inputs.

    The conditional Gaussian process is a Gaussian process distribution which is conditioned over a pair of observation X_cond and Y_cond.

    .. math::
       Y \\sim \\mathcal{N}(Y| K_{*c}K_{cc}^{-1}(Y_C - g(X_c)) + g(X), K_{**} - K_{*c}K_{cc}^{-1}K_{*c}^\\top)

    where :math:`g` is the mean function, :math:`K_{**}` is the covariance matrix over :math:`X`, :math:`K_{*c}` is the cross covariance matrix
    between :math:`X` and :math:`X_{c}` and :math:`K_{cc}` is the covariance matrix over :math:`X_{c}`.

    :param X: the input variables on which the random variables are conditioned.
    :type X: Variable
    :param X_cond: the input variables on which the output variables *Y_Cond* are conditioned.
    :type X_cond: Variable
    :param Y_cond: the output variables on which the random variables are conditioned.
    :type Y_cond: Variable
    :param kernel: the kernel of Gaussian process.
    :type kernel: Kernel
    :param mean: the mean of Gaussian process.
    :type mean: Variable
    :param mean_cond: the mean of the conditional output variable under the same mean function.
    :type mean_cond: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, X, X_cond, Y_cond, kernel, mean=None, mean_cond=None,
                 rand_gen=None, dtype=None, ctx=None):
        if (mean is None) and (mean_cond is not None):
            raise ModelSpecificationError("The argument mean and mean_cond need to be both specified.")
        inputs = [('X', X), ('X_cond', X_cond), ('Y_cond', Y_cond)] + \
            [(k, v) for k, v in kernel.parameters.items()]
        input_names = [k for k, _ in inputs]
        if mean is not None:
            inputs.append(('mean', mean))
            input_names.append('mean')
            self._has_mean = True
        else:
            self._has_mean = False
        if mean_cond is not None:
            inputs.append(('mean_cond', mean_cond))
            input_names.append('mean_cond')
            self._has_mean_cond = True
        else:
            self._has_mean_cond = False
        output_names = ['random_variable']
        super(ConditionalGaussianProcess, self).__init__(
            inputs=inputs, outputs=None, input_names=input_names,
            output_names=output_names, rand_gen=rand_gen, dtype=dtype,
            ctx=ctx)
        self.kernel = kernel

    @property
    def has_mean(self):
        return self._has_mean

    @staticmethod
    def define_variable(X, X_cond, Y_cond, kernel, shape=None, mean=None,
                        mean_cond=None, rand_gen=None, minibatch_ratio=1.,
                        dtype=None, ctx=None):
        """
        Creates and returns a set of random variable drawn from a Gaussian process.

        :param X: the input variables on which the random variables are conditioned.
        :type X: Variable
        :param X_cond: the input variables on which the output variables *Y_Cond* are conditioned.
        :type X_cond: Variable
        :param Y_cond: the output variables on which the random variables are conditioned.
        :type Y_cond: Variable
        :param kernel: the kernel of Gaussian process.
        :type kernel: Kernel
        :param shape: the shape of the random variable(s) (the default shape is the same shape as *X* but the last dimension is changed to one.)
        :type shape: tuple or [tuple]
        :param mean: the mean of Gaussian process.
        :type mean: Variable
        :param mean_cond: the mean of the conditional output variable under the same mean function.
        :type mean_cond: Variable
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        """
        gp = ConditionalGaussianProcess(
            X=X, X_cond=X_cond, Y_cond=Y_cond, kernel=kernel, mean=mean,
            mean_cond=mean_cond, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        gp.outputs = [('random_variable',
                      Variable(value=gp, shape=X.shape[:-1] + (1,) if
                               shape is None else shape))]
        return gp.random_variable

    def log_pdf_impl(self, X, X_cond, Y_cond, random_variable, F=None,
                     **kernel_params):
        """
        Computes the logarithm of the probability density function (PDF) of the conditional Gaussian process.

        .. math::
           \\log p(Y| X_c, Y_c, X) = \\log \\mathcal{N}(Y| K_{*c}K_{cc}^{-1}(Y_C - g(X_c)) + g(X), K_{**} - K_{*c}K_{cc}^{-1}K_{*c}^\\top)

        where :math:`g` is the mean function, :math:`K_{**}` is the covariance matrix over :math:`X`, :math:`K_{*c}` is the cross covariance matrix
        between :math:`X` and :math:`X_{c}` and :math:`K_{cc}` is the covariance matrix over :math:`X_{c}`.

        :param X: the input variables on which the random variables are conditioned.
        :type X: MXNet NDArray or MXNet Symbol
        :param X_cond: the input variables on which the output variables *Y_Cond* are conditioned.
        :type X_cond: MXNet NDArray or MXNet Symbol
        :param Y_cond: the output variables on which the random variables are conditioned.
        :type Y_cond: MXNet NDArray or MXNet Symbol
        :param random_variable: the random_variable of which log-PDF is computed.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :param **kernel_params: the set of kernel parameters, provided as keyword arguments.
        :type **kernel_params: {str: MXNet NDArray or MXNet Symbol}
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        if self._has_mean:
            mean = kernel_params['mean']
            del kernel_params['mean']
        if self._has_mean_cond:
            mean_cond = kernel_params['mean_cond']
            del kernel_params['mean_cond']
        D = random_variable.shape[-1]
        F = get_default_MXNet_mode() if F is None else F
        K = self.kernel.K(F, X, **kernel_params)
        Kc = self.kernel.K(F, X_cond, X, **kernel_params)
        Kcc = self.kernel.K(F, X_cond, **kernel_params)
        Lcc = F.linalg.potrf(Kcc)
        LccInvKc = F.linalg.trsm(Lcc, Kc)
        cov = K - F.linalg.syrk(LccInvKc, transpose=True)
        L = F.linalg.potrf(cov)
        if self._has_mean:
            random_variable = random_variable - mean
        if self._has_mean_cond:
            Y_cond = Y_cond - mean_cond
        LccInvY = F.linalg.trsm(Lcc, Y_cond)
        rv_mean = F.linalg.gemm2(LccInvKc, LccInvY, True, False)
        LinvY = F.sum(F.linalg.trsm(L, random_variable - rv_mean), axis=-1)
        logdet_l = F.linalg.sumlogdiag(F.abs(L))

        return (- logdet_l * D - F.sum(F.square(LinvY) + np.log(2. * np.pi),
                axis=-1) / 2) * self.log_pdf_scaling

    def draw_samples_impl(self, X, X_cond, Y_cond, rv_shape, num_samples=1,
                          F=None, **kernel_params):
        """
        Draw a number of samples from the conditional Gaussian process.

        :param X: the input variables on which the random variables are conditioned.
        :type X: MXNet NDArray or MXNet Symbol
        :param X_cond: the input variables on which the output variables *Y_Cond* are conditioned.
        :type X_cond: MXNet NDArray or MXNet Symbol
        :param Y_cond: the output variables on which the random variables are conditioned.
        :type Y_cond: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :type num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :param **kernel_params: the set of kernel parameters, provided as keyword arguments.
        :type **kernel_params: {str: MXNet NDArray or MXNet Symbol}
        :returns: a set samples of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        if self._has_mean:
            mean = kernel_params['mean']
            del kernel_params['mean']
        if self._has_mean_cond:
            mean_cond = kernel_params['mean_cond']
            del kernel_params['mean_cond']
        F = get_default_MXNet_mode() if F is None else F
        K = self.kernel.K(F, X, **kernel_params)
        Kc = self.kernel.K(F, X_cond, X, **kernel_params)
        Kcc = self.kernel.K(F, X_cond, **kernel_params)
        Lcc = F.linalg.potrf(Kcc)
        LccInvKc = F.linalg.trsm(Lcc, Kc)
        cov = K - F.linalg.syrk(LccInvKc, transpose=True)
        L = F.linalg.potrf(cov)
        if self._has_mean_cond:
            Y_cond = Y_cond - mean_cond
        LccInvY = F.linalg.trsm(Lcc, Y_cond)
        rv_mean = F.linalg.gemm2(LccInvKc, LccInvY, True, False)

        out_shape = (num_samples,) + rv_shape

        die = self._rand_gen.sample_normal(
            shape=out_shape, dtype=self.dtype, ctx=self.ctx)
        rv = F.linalg.trmm(L, die) + rv_mean
        if self._has_mean:
            rv = rv + mean
        return rv

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for a conditional Gaussian process distribution.
        """
        replicant = super(ConditionalGaussianProcess,
                          self).replicate_self(attribute_map)
        replicant._has_mean = self._has_mean
        replicant._has_mean_cond = self._has_mean_cond
        replicant.kernel = self.kernel.replicate_self(attribute_map)
        return replicant
