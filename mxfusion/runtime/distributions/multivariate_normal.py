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
from mxnet.ndarray.linalg import sumlogdiag, trsm
import numpy as np

from ...common.exceptions import InferenceError
from .distribution import RuntimDistribution
from ...components.variables.runtime_variable import get_variable_shape


class MultivariateNormal(RuntimDistribution):
    """
    Multi-dimensional normal distribution. Can represent a number of independent multivariate normal distributions.
    """
    def __init__(self, mean, covariance=None):
        """
        :param mean: Mean of the normal distribution. Shape: (n_samples, n_outputs, n_dim)
        :type mean: MXNet NDArray
        :param covariance: Covariance matrix of the distribution. Shape: (n_samples, n_outputs, n_dim, n_dim)
        :type covariance: MXNet NDArray
        """
        super(MultivariateNormal, self).__init__()

        # if mean.ndim != 3:
        #     raise ValueError('Mean should have 3 dimensions. It has {}'.format(mean.ndim))
        #
        # if covariance.ndim != 4:
        #     raise ValueError('Covariance should have 4 dimensions. It has {}'.format(covariance.ndim))

        if covariance is None or mean.shape != covariance.shape[:-1] or covariance.shape[-2]!=covariance.shape[-1]:
            raise InferenceError('Mean and covariance shapes inconsistent. Mean shape: {}. Covariance shape: {}'.format(
                mean.shape, covariance.shape))

        self.mean = mean
        self._covariance = covariance
        self._cholesky = None

    @staticmethod
    def from_cholesky(mean, cholesky):
        if mean.shape != cholesky.shape[:-1]:
            raise InferenceError('Mean and Cholesky shapes inconsistent. Mean shape: {}. Covariance shape: {}'.format(
                mean.shape, cholesky.shape))
        dist = MultivariateNormal(mean)
        dist._cholesky = cholesky
        return dist

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        self._cholesky = None
        self._covariance = value

    @property
    def cholesky(self):
        if self._cholesky is None:
            self._cholesky = mx.nd.linalg.potrf(self.covariance)
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

        if random_variable.shape[0] != self.mean.shape[0]:
            raise InferenceError('Number of samples in random variable must {}'.format(self.mean.shape[0]))

        F = mx.nd
        D = random_variable.shape[-1]
        L = self.cholesky
        Y = random_variable - self.mean
        LinvY = trsm(L, mx.nd.expand_dims(Y, -1))
        logdet_l = sumlogdiag(F.abs(L))
        log_pdf = - logdet_l * D - F.sum(F.sum(F.square(LinvY) + np.log(2. * np.pi), axis=-1), axis=-1) / 2
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

        num_out_samples = max(num_samples, self.mean.shape[0])
        out_shape = (num_out_samples,) + get_variable_shape(mx.nd, self.mean) + (1,)
        lmat = self.cholesky
        epsilon = mx.nd.random.normal(shape=out_shape, dtype=self.mean.dtype, ctx=self.mean.context)

        if lmat.shape[0] != num_out_samples:
            lmat = mx.nd.broadcast_axis(lmat, axis=0, size=num_out_samples)
        lmat_eps = mx.nd.linalg.trmm(lmat, epsilon)
        return mx.nd.broadcast_add(lmat_eps.sum(-1), self.mean)

    def kl_divergence(self, other):
        """
        Computes KL(self, other). Both distributions must have the same shape.

        :param other: Another MultiVariateNormal distribution
        :type other: MultiVariateNormal
        :rtypes: MXNet NDArray
        """

        if not isinstance(other, MultivariateNormal):
            raise InferenceError('KL divergence for MultiVariateNormal only implemented for another MultiVariateNormal, not '
                            'a {} object.'.format(type(other)))

        if self.mean.shape != other.mean.shape or self.cholesky.shape != other.cholesky.shape:
            raise InferenceError('This distribution covariance matrix has shape {}. Other distribution covariance matrix '
                             'has shape {}. They should be the same'.format(self.cholesky.shape,
                                                                            other.cholesky.shape))

        # Notation: D is number of output dimension, N is number of samples and M is number of input dimensions.
        L_1 = self.cholesky   # N x D x M x M
        L_2 = other.cholesky  # N x D x M x M

        mean_diff = self.mean - other.mean  # N x D X M
        mean_diff = mx.nd.expand_dims(mean_diff, -1)

        M = self.mean.shape[-1]

        LinvLs = mx.nd.linalg.trsm(L_2, L_1)  # N x D x M x M
        Linvmu = mx.nd.linalg.trsm(L_2, mean_diff)  # N x D x M x M

        return -M/2 + sumlogdiag(L_2) - sumlogdiag(L_1).sum() + \
            mx.nd.square(LinvLs).sum(-1).sum(-1)/2 + \
            mx.nd.square(Linvmu).sum(-1).sum(-1)/2
