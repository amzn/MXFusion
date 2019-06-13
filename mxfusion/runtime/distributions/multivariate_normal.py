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
import mxnet as mx
from mxnet.ndarray import expand_dims, abs, sum, square, diag, broadcast_axis
from mxnet.ndarray.linalg import sumlogdiag, trsm, syrk, potrf
import numpy as np

from ...common.exceptions import InferenceError
from .distribution import DistributionRuntime


class MultivariateNormalRuntime(DistributionRuntime):
    """
    Multi-dimensional normal distribution. Can represent a number of independent multivariate normal distributions.
    """
    def __init__(self, mean, covariance=None, cholesky_factor=None):
        """
        :param mean: Mean of the normal distribution. Shape: (n_samples, n_outputs, n_dim)
        :type mean: MXNet NDArray
        :param covariance: Covariance matrix of the distribution. Shape: (n_samples, n_outputs, n_dim, n_dim)
        :type covariance: MXNet NDArray
        """
        super(MultivariateNormalRuntime, self).__init__()

        if covariance is None or mean.shape != covariance.shape[:-1] or covariance.shape[-2]!=covariance.shape[-1]:
            raise InferenceError('Mean and covariance shapes inconsistent. Mean shape: {}. Covariance shape: {}'.format(
                mean.shape, covariance.shape))

        if covariance is None and cholesky_factor is None:
            raise InferenceError('Either covariance or cholesky_factor needs to be specified.')

        self.mean = mean
        self._covariance = covariance
        self._cholesky = cholesky_factor

    @property
    def variance(self):
        return diag(self.covariance, axis1=-2, axis2=-1)

    @property
    def covariance(self):
        if self._covariance is None:
            self._covariance = syrk(self._cholesky)
        return self._covariance

    @property
    def cholesky_factor(self):
        if self._cholesky is None:
            self._cholesky = potrf(self._covariance)
        return self._cholesky

    def log_pdf(self, random_variable):
        """
        Computes the logarithm of the probability density function (PDF) of the multi-variate normal distribution.

        The input should have the same number of samples as are in the mean and covariance of the distribution

        :param random_variable: the random variable of the normal distribution.
        :type random_variable: MXNet NDArray
        :return: Evaluation of log pdf function. Shape (n_samples, n_outputs) where n_samples is the greater of the
                 number of samples in the distribution definition and the random variable input
        :rtypes: MXNet NDArray
        """

        if random_variable.shape[1:] != self.mean.shape[1:]:
            raise InferenceError('Non-sample dimension of random variable has shape {} but expected {}'.format(
                random_variable.shape[1:], self.mean.shape[1:]))

        D = self.mean.shape[-1]
        L = self.cholesky_factor
        Y = random_variable - self.mean
        if Y.shape[0] < L.shape[0]:
            Y = broadcast_axis(Y, axis=0, size=L.shape[0])
        elif Y.shape[0] > L.shape[0]:
            L = broadcast_axis(L, axis=0, size=Y.shape[0])
        LinvY = trsm(L, expand_dims(Y, -1)).sum(-1)
        log_pdf = - sumlogdiag(abs(L)) - D*np.log(2*np.pi)/2 - sum(square(LinvY), axis=-1)/2
        return log_pdf

    def draw_samples(self, num_samples=1):
        """
        Draw a number of samples from the normal distribution.

        :param num_samples: the number of drawn samples (default: 1).
        :type num_samples: int
        :rtypes: MXNet NDArray
        """

        if (self.mean.shape[0] != 1) and (num_samples != 1) and (num_samples != self.mean.shape[0]):
            raise InferenceError('Number of samples must be 1 or {}'.format(self.mean.shape[0]))

        out_shape = (num_samples,) + self.mean.shape[1:]
        L = self.cholesky_factor
        epsilon = mx.nd.random.normal(shape=out_shape+(1,), dtype=self.mean.dtype, ctx=self.mean.context)

        if L.shape[0] != num_samples:
            L = mx.nd.broadcast_axis(L, axis=0, size=num_samples)
        lmat_eps = mx.nd.linalg.trmm(L, epsilon)
        return lmat_eps.sum(-1)+self.mean

    def kl_divergence(self, other):
        """
        Computes KL(self, other). Both distributions must have the same shape.

        :param other: Another MultivariateNormalRuntime distribution
        :type other: MultivariateNormalRuntime
        :rtypes: MXNet NDArray
        """
        # Notation: D is number of output dimension, N is number of samples and M is number of input dimensions.
        L_1 = self.cholesky_factor   # N x D x M x M
        L_2 = other.cholesky_factor  # N x D x M x M

        if not isinstance(other, MultivariateNormalRuntime):
            raise InferenceError('KL divergence for MultivariateNormalRuntime only implemented for another MultivariateNormalRuntime, not '
                            'a {} object.'.format(type(other)))

        if self.mean.shape != other.mean.shape or L_1.shape != L_2.shape:
            raise InferenceError('This distribution covariance matrix has shape {}. Other distribution covariance matrix '
                             'has shape {}. They should be the same'.format(L_1.shape, L_2.shape))

        mean_diff = self.mean - other.mean  # N x D X M
        mean_diff = expand_dims(mean_diff, -1)

        M = self.mean.shape[-1]

        LinvLs = mx.nd.linalg.trsm(L_2, L_1)  # N x D x M x M
        Linvmu = mx.nd.linalg.trsm(L_2, mean_diff)  # N x D x M x M

        return -M/2 + sumlogdiag(L_2) - sumlogdiag(L_1) + square(LinvLs).reshape(*(L_1.shape[:-2]+(-1,))).sum(-1)/2 + \
            square(Linvmu).sum(-1).sum(-1)/2
