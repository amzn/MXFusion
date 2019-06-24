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


import pytest
import mxnet as mx
import numpy as np
from mxfusion.runtime.distributions import MultivariateNormalRuntime, MultivariateNormalMeanPrecisionRuntime
from mxfusion.common.exceptions import InferenceError
np.random.seed(0)
mx.random.seed(0)


@pytest.mark.usefixtures("set_seed")
class TestMultivariateNormalRuntimeDistribution(object):

    @pytest.mark.parametrize("dtype, mean, cov, rv", [
        (np.float64, np.random.rand(1,3), np.random.rand(1,3,3), np.random.rand(1,3)),
        (np.float64, np.random.rand(4,3), np.random.rand(4,3,3), np.random.rand(4,3)),
        (np.float64, np.random.rand(4,3), np.random.rand(4,3,3), np.random.rand(1,3)),
        (np.float32, np.random.rand(4,3), np.random.rand(4,3,3), np.random.rand(4,3))
        ])
    def test_log_pdf(self, dtype, mean, cov, rv):
        cov = (np.expand_dims(cov, -2)*np.expand_dims(cov, -3)).sum(-1)

        from scipy.stats import multivariate_normal
        if mean.ndim > 1:
            if rv.shape[0]<mean.shape[0]:
                rv_r = np.repeat(rv, mean.shape[0], 0)
            else:
                rv_r = rv
            log_pdf_np = np.array([multivariate_normal.logpdf(rv_r[i], mean[i], cov[i]) for i in range(mean.shape[0])])
        else:
            log_pdf_np = multivariate_normal.logpdf(rv, mean, cov)

        mean_mx = mx.nd.array(mean, dtype=dtype)
        cov_mx = mx.nd.array(cov, dtype=dtype)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        mvn_dist = MultivariateNormalRuntime(mean=mean_mx, covariance=cov_mx)
        log_pdf_rt = mvn_dist.log_pdf(rv_mx)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-3, 1e-4
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)

    def test_draw_samples(self):
        np.random.seed(0)
        mx.random.seed(0)
        num_samples = 1000

        mean_mx = mx.nd.array(np.random.rand(1,2), dtype='float64')
        cov = np.random.rand(1,2,2)
        cov = (np.expand_dims(cov, -2)*np.expand_dims(cov, -3)).sum(-1)
        cov_mx = mx.nd.array(cov, dtype='float64')
        mvn_dist = MultivariateNormalRuntime(mean=mean_mx, covariance=cov_mx)
        samples = mvn_dist.draw_samples(num_samples).asnumpy()
        mean = samples.mean(0)
        cov = samples.T.dot(samples)/num_samples - mean[:, None]*mean[None, :]
        rtol, atol = 1e-1, 1e-1
        assert np.allclose(mean, mvn_dist.mean.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(samples.var(0), mvn_dist.variance.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(cov, mvn_dist.covariance.asnumpy(), rtol=rtol, atol=atol)

    def test_draw_samples_broadcast(self):
        num_samples = 10

        mean_mx = mx.nd.array(np.random.rand(num_samples,2), dtype='float64')
        cov = np.random.rand(num_samples,2,2)
        cov = (np.expand_dims(cov, -2)*np.expand_dims(cov, -3)).sum(-1)
        cov_mx = mx.nd.array(cov, dtype='float64')
        mvn_dist = MultivariateNormalRuntime(mean=mean_mx, covariance=cov_mx)
        samples = mvn_dist.draw_samples(num_samples).asnumpy()
        assert samples.shape == (num_samples, 2)


def make_covariance(n_samples, n_out, n_dims):
    L = np.random.rand(n_dims, n_dims)
    cov = L.T.dot(L) + np.eye(n_dims)
    return mx.nd.array(np.tile(cov[None, None, :, :], (n_samples, n_out, 1, 1)))


def test_log_pdf():
    mean = mx.nd.zeros((10, 2, 3))
    covariance = make_covariance(10, 2, 3)
    mvn = MultivariateNormalRuntime(mean, covariance)

    rv = mx.nd.random.normal(shape=(10, 2, 3))
    result = mvn.log_pdf(rv)
    assert result.shape == (10, 2)


@pytest.mark.parametrize(argnames="dist_shape, n_samples, expected_shape",
                         argvalues=(((10, 2, 3), 10, (10, 2, 3)),
                                    ((1, 2, 3),  10, (10, 2, 3)),
                                    ((10, 2, 3), 10,  (10, 2, 3))))
def test_draw_samples(dist_shape, n_samples, expected_shape):
    mean = mx.nd.zeros(dist_shape)
    covariance = mx.nd.tile(mx.nd.eye(dist_shape[-1]), dist_shape[:2] + (1, 1))
    mvn = MultivariateNormalRuntime(mean, covariance)

    result = mvn.draw_samples(n_samples)
    assert result.shape == expected_shape


def test_kl_zero_between_identical_distributions():
    cov_1 = mx.nd.tile(mx.nd.eye(3), (10, 2, 1, 1))
    cov_2 = mx.nd.tile(mx.nd.eye(3), (10, 2, 1, 1))

    mvn_1 = MultivariateNormalRuntime(mx.nd.zeros((10, 2, 3)), cov_1)
    mvn_2 = MultivariateNormalRuntime(mx.nd.zeros((10, 2, 3)), cov_2)

    assert mvn_1.kl_divergence(mvn_2).sum().asnumpy() == 0


def test_kl_different_means():
    cov_1 = mx.nd.tile(mx.nd.eye(3), (10, 2, 1, 1))
    cov_2 = mx.nd.tile(mx.nd.eye(3), (10, 2, 1, 1))

    mvn_1 = MultivariateNormalRuntime(mx.nd.zeros((10, 2, 3)), cov_1)
    mvn_2 = MultivariateNormalRuntime(mx.nd.ones((10, 2, 3)), cov_2)

    assert mvn_1.kl_divergence(mvn_2).sum().asnumpy() == 30


@pytest.mark.parametrize(argnames=("mean, covariance"),
                         argvalues=((mx.nd.random.randn(5, 1, 2), mx.nd.random.randn(1, 1, 2, 2)),
                                    (mx.nd.random.randn(1, 5, 2), mx.nd.random.randn(1, 1, 2, 2)),
                                    (mx.nd.random.randn(5, 2), mx.nd.random.randn(5, 2, 3))))
def test_initialization_with_wrong_sizes(mean, covariance):
    with pytest.raises(InferenceError):
        MultivariateNormalRuntime(mean, covariance)


@pytest.mark.parametrize(argnames=("mean, covariance, rv"),
                         argvalues=((mx.nd.random.randn(5, 1, 2), make_covariance(5, 1, 2), mx.nd.random.randn(1, 2)),))
def test_log_pdf_wrong_shape(mean, covariance, rv):
    mvn = MultivariateNormalRuntime(mean, covariance)
    with pytest.raises(InferenceError):
        mvn.log_pdf(rv)


def test_kl_inconsistent_shapes():
    cov_1 = mx.nd.tile(mx.nd.eye(3), (1, 2, 1, 1))
    cov_2 = mx.nd.tile(mx.nd.eye(3), (10, 2, 1, 1))

    mvn_1 = MultivariateNormalRuntime(mx.nd.zeros((1, 2, 3)), cov_1)
    mvn_2 = MultivariateNormalRuntime(mx.nd.ones((10, 2, 3)), cov_2)

    with pytest.raises(InferenceError):
        mvn_1.kl_divergence(mvn_2)


@pytest.mark.usefixtures("set_seed")
class TestMultivariateNormalMeanPrecisionRuntimeDistribution(object):

    @pytest.mark.parametrize("dtype, mean, cov, rv", [
        (np.float64, np.random.rand(1,3), np.random.rand(1,3,3), np.random.rand(1,3)),
        (np.float64, np.random.rand(4,3), np.random.rand(4,3,3), np.random.rand(4,3)),
        (np.float64, np.random.rand(4,3), np.random.rand(4,3,3), np.random.rand(1,3)),
        (np.float32, np.random.rand(4,3), np.random.rand(4,3,3), np.random.rand(4,3))
        ])
    def test_log_pdf(self, dtype, mean, cov, rv):
        cov = (np.expand_dims(cov, -2)*np.expand_dims(cov, -3)).sum(-1)

        from scipy.stats import multivariate_normal
        from scipy.linalg import cholesky
        from scipy.linalg.lapack import dpotri
        prec = np.array([dpotri(cholesky(cov[i], lower=True), lower=True)[0] for i in range(cov.shape[0])])
        if mean.ndim > 1:
            if rv.shape[0]<mean.shape[0]:
                rv_r = np.repeat(rv, mean.shape[0], 0)
            else:
                rv_r = rv
            log_pdf_np = np.array([multivariate_normal.logpdf(rv_r[i], mean[i], cov[i]) for i in range(mean.shape[0])])
        else:
            log_pdf_np = multivariate_normal.logpdf(rv, mean, cov)

        mean_mx = mx.nd.array(mean, dtype=dtype)
        prec_mx = mx.nd.array(prec, dtype=dtype)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        mvn_dist = MultivariateNormalMeanPrecisionRuntime(mean=mean_mx, precision=prec_mx)
        log_pdf_rt = mvn_dist.log_pdf(rv_mx)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-3, 1e-4
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)

    def test_draw_samples(self):
        from scipy.linalg import cholesky
        from scipy.linalg.lapack import dpotri

        np.random.seed(0)
        mx.random.seed(0)
        num_samples = 1000

        mean_mx = mx.nd.array(np.random.rand(1,2), dtype='float64')
        cov = np.random.rand(1,2,2)
        cov = (np.expand_dims(cov, -2)*np.expand_dims(cov, -3)).sum(-1)
        prec = np.array([dpotri(cholesky(cov[i], lower=True), lower=True)[0] for i in range(cov.shape[0])])
        prec_mx = mx.nd.array(prec, dtype='float64')
        mvn_dist = MultivariateNormalMeanPrecisionRuntime(mean=mean_mx, precision=prec_mx)
        samples = mvn_dist.draw_samples(num_samples).asnumpy()
        mean = samples.mean(0)
        cov = samples.T.dot(samples)/num_samples - mean[:, None]*mean[None, :]
        rtol, atol = 1e-1, 1e-1
        assert np.allclose(mean, mvn_dist.mean.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(samples.var(0), mvn_dist.variance.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(cov, mvn_dist.covariance.asnumpy(), rtol=rtol, atol=atol)

    def test_draw_samples_broadcast(self):
        from scipy.linalg import cholesky
        from scipy.linalg.lapack import dpotri
        np.random.seed(0)
        mx.random.seed(0)
        num_samples = 10

        mean_mx = mx.nd.array(np.random.rand(num_samples,2), dtype='float64')
        cov = np.random.rand(num_samples,2,2)
        cov = (np.expand_dims(cov, -2)*np.expand_dims(cov, -3)).sum(-1)
        prec = np.array([dpotri(cholesky(cov[i], lower=True), lower=True)[0] for i in range(cov.shape[0])])
        prec_mx = mx.nd.array(prec, dtype='float64')
        mvn_dist = MultivariateNormalMeanPrecisionRuntime(mean=mean_mx, precision=prec_mx)
        samples = mvn_dist.draw_samples(num_samples).asnumpy()
        assert samples.shape == (num_samples, 2)
