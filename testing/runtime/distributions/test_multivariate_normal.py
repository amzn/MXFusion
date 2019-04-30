import mxnet as mx
from mxfusion.runtime.distributions.multivariate_normal import MultivariateNormal
from mxfusion.common.exceptions import InferenceError
import pytest
import numpy as np


def make_covariance(n_samples, n_out, n_dims):
    L = np.random.rand(n_dims, n_dims)
    cov = L.T.dot(L) + np.eye(n_dims)
    return mx.nd.array(np.tile(cov[None, None, :, :], (n_samples, n_out, 1, 1)))


def test_log_pdf():
    mean = mx.nd.zeros((10, 2, 3))
    covariance = make_covariance(10, 2, 3)
    mvn = MultivariateNormal(mean, covariance)

    rv = mx.nd.random.normal(shape=(10, 2, 3))
    result = mvn.log_pdf(rv)
    assert result.shape == (10, 2)


@pytest.mark.parametrize(argnames="dist_shape, n_samples, expected_shape",
                         argvalues=(((10, 2, 3), 10, (10, 2, 3)),
                                    ((1, 2, 3),  10, (10, 2, 3)),
                                    ((10, 2, 3), 1,  (10, 2, 3))))
def test_draw_samples(dist_shape, n_samples, expected_shape):
    mean = mx.nd.zeros(dist_shape)
    covariance = mx.nd.tile(mx.nd.eye(dist_shape[-1]), dist_shape[:2] + (1, 1))
    mvn = MultivariateNormal(mean, covariance)

    result = mvn.draw_samples(n_samples)
    assert result.shape == expected_shape


def test_kl_zero_between_identical_distributions():
    cov_1 = mx.nd.tile(mx.nd.eye(3), (10, 2, 1, 1))
    cov_2 = mx.nd.tile(mx.nd.eye(3), (10, 2, 1, 1))

    mvn_1 = MultivariateNormal(mx.nd.zeros((10, 2, 3)), cov_1)
    mvn_2 = MultivariateNormal(mx.nd.zeros((10, 2, 3)), cov_2)

    assert mvn_1.kl_divergence(mvn_2).sum().asnumpy() == 0


def test_kl_different_means():
    cov_1 = mx.nd.tile(mx.nd.eye(3), (10, 2, 1, 1))
    cov_2 = mx.nd.tile(mx.nd.eye(3), (10, 2, 1, 1))

    mvn_1 = MultivariateNormal(mx.nd.zeros((10, 2, 3)), cov_1)
    mvn_2 = MultivariateNormal(mx.nd.ones((10, 2, 3)), cov_2)

    assert mvn_1.kl_divergence(mvn_2).sum().asnumpy() == 30


@pytest.mark.parametrize(argnames=("mean, covariance"),
                         argvalues=((mx.nd.random.randn(5, 1, 2), mx.nd.random.randn(1, 1, 2, 2)),
                                    (mx.nd.random.randn(1, 5, 2), mx.nd.random.randn(1, 1, 2, 2)),
                                    (mx.nd.random.randn(5, 2), mx.nd.random.randn(5, 2, 3))))
def test_initialization_with_wrong_sizes(mean, covariance):
    with pytest.raises(InferenceError):
        MultivariateNormal(mean, covariance)


@pytest.mark.parametrize(argnames=("mean, covariance, rv"),
                         argvalues=((mx.nd.random.randn(5, 1, 2), make_covariance(5, 1, 2), mx.nd.random.randn(3, 1, 2)),
                                    (mx.nd.random.randn(5, 1, 2), make_covariance(5, 1, 2), mx.nd.random.randn(1, 2))))
def test_log_pdf_wrong_shape(mean, covariance, rv):
    mvn = MultivariateNormal(mean, covariance)
    with pytest.raises(InferenceError):
        mvn.log_pdf(rv)


def test_kl_inconsistent_shapes():
    cov_1 = mx.nd.tile(mx.nd.eye(3), (1, 2, 1, 1))
    cov_2 = mx.nd.tile(mx.nd.eye(3), (10, 2, 1, 1))

    mvn_1 = MultivariateNormal(mx.nd.zeros((1, 2, 3)), cov_1)
    mvn_2 = MultivariateNormal(mx.nd.ones((10, 2, 3)), cov_2)

    with pytest.raises(InferenceError):
        mvn_1.kl_divergence(mvn_2)
