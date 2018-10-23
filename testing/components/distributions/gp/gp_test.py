import pytest
import mxnet as mx
import numpy as np
from mxfusion.models import Model
from mxfusion.components.variables.runtime_variable import array_has_samples, get_num_samples
from mxfusion.components.distributions import GaussianProcess
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.components import Variable
from mxfusion.util.testutils import prepare_mxnet_array
from mxfusion.util.testutils import MockMXNetRandomGenerator
from scipy.stats import multivariate_normal
import matplotlib
matplotlib.use('Agg')
import GPy


@pytest.mark.usefixtures("set_seed")
class TestGaussianProcessDistribution(object):

    @pytest.mark.parametrize("dtype, X, X_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, rv, rv_isSamples, num_samples", [
        (np.float64, np.random.rand(5,2), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, np.random.rand(3,5,1), True, 3),
        (np.float64, np.random.rand(3,5,2), True, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, np.random.rand(5,1), False, 3),
        (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,2)+0.1, True, np.random.rand(3,1)+0.1, True, np.random.rand(3,5,1), True, 3),
        (np.float64, np.random.rand(5,2), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, np.random.rand(5,1), False, 1),
        ])
    def test_log_pdf(self, dtype, X, X_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples,
                        rv, rv_isSamples, num_samples):
        X_mx = prepare_mxnet_array(X, X_isSamples, dtype)
        rbf_lengthscale_mx = prepare_mxnet_array(rbf_lengthscale, rbf_lengthscale_isSamples, dtype)
        rbf_variance_mx = prepare_mxnet_array(rbf_variance, rbf_variance_isSamples, dtype)
        rv_mx = prepare_mxnet_array(rv, rv_isSamples, dtype)
        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape

        rbf = RBF(2, True, 1., 1., 'rbf', None, dtype)
        X_var = Variable(shape=(5,2))
        gp = GaussianProcess.define_variable(X=X_var, kernel=rbf, shape=rv_shape, dtype=dtype).factor

        variables = {gp.X.uuid: X_mx, gp.rbf_lengthscale.uuid: rbf_lengthscale_mx, gp.rbf_variance.uuid: rbf_variance_mx, gp.random_variable.uuid: rv_mx}
        log_pdf_rt = gp.log_pdf(F=mx.nd, variables=variables).asnumpy()

        log_pdf_np = []
        for i in range(num_samples):
            X_i = X[i] if X_isSamples else X
            lengthscale_i = rbf_lengthscale[i] if rbf_lengthscale_isSamples else rbf_lengthscale
            variance_i = rbf_variance[i] if rbf_variance_isSamples else rbf_variance
            rv_i = rv[i] if rv_isSamples else rv
            rbf_np = GPy.kern.RBF(input_dim=2, ARD=True)
            rbf_np.lengthscale = lengthscale_i
            rbf_np.variance = variance_i
            K_np = rbf_np.K(X_i)
            log_pdf_np.append(multivariate_normal.logpdf(rv_i[:,0], mean=None, cov=K_np))
        log_pdf_np = np.array(log_pdf_np)
        isSamples_any = any([X_isSamples, rbf_lengthscale_isSamples, rbf_variance_isSamples, rv_isSamples])
        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == isSamples_any
        if isSamples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        assert np.allclose(log_pdf_np, log_pdf_rt)

    @pytest.mark.parametrize("dtype, X, X_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, rv_shape, num_samples", [
        (np.float64, np.random.rand(5,2), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, (5,1), 3),
        (np.float64, np.random.rand(3,5,2), True, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, (5,1), 3),
        (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,2)+0.1, True, np.random.rand(3,1)+0.1, True, (5,1), 3),
        (np.float64, np.random.rand(5,2), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, (5,1), 1),
        ])
    def test_draw_samples(self, dtype, X, X_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples,
                        rv_shape, num_samples):
        X_mx = prepare_mxnet_array(X, X_isSamples, dtype)
        rbf_lengthscale_mx = prepare_mxnet_array(rbf_lengthscale, rbf_lengthscale_isSamples, dtype)
        rbf_variance_mx = prepare_mxnet_array(rbf_variance, rbf_variance_isSamples, dtype)

        rand = np.random.randn(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        rbf = RBF(2, True, 1., 1., 'rbf', None, dtype)
        X_var = Variable(shape=(5,2))
        gp = GaussianProcess.define_variable(X=X_var, kernel=rbf, shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor

        variables = {gp.X.uuid: X_mx, gp.rbf_lengthscale.uuid: rbf_lengthscale_mx, gp.rbf_variance.uuid: rbf_variance_mx}
        samples_rt = gp.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples).asnumpy()

        samples_np = []
        for i in range(num_samples):
            X_i = X[i] if X_isSamples else X
            lengthscale_i = rbf_lengthscale[i] if rbf_lengthscale_isSamples else rbf_lengthscale
            variance_i = rbf_variance[i] if rbf_variance_isSamples else rbf_variance
            rand_i = rand[i]
            rbf_np = GPy.kern.RBF(input_dim=2, ARD=True)
            rbf_np.lengthscale = lengthscale_i
            rbf_np.variance = variance_i
            K_np = rbf_np.K(X_i)
            L_np = np.linalg.cholesky(K_np)
            sample_np = L_np.dot(rand_i)
            samples_np.append(sample_np)
        samples_np = np.array(samples_np)

        assert np.issubdtype(samples_rt.dtype, dtype)
        assert get_num_samples(mx.nd, samples_rt) == num_samples
        assert np.allclose(samples_np, samples_rt)


    @pytest.mark.parametrize("dtype, X, X_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, rv, rv_isSamples, num_samples", [
        (np.float64, np.random.rand(5,2), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, np.random.rand(3,5,1), True, 3),
        ])
    def test_clone_gp(self, dtype, X, X_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples,
                        rv, rv_isSamples, num_samples):
        X_mx = prepare_mxnet_array(X, X_isSamples, dtype)
        rbf_lengthscale_mx = prepare_mxnet_array(rbf_lengthscale, rbf_lengthscale_isSamples, dtype)
        rbf_variance_mx = prepare_mxnet_array(rbf_variance, rbf_variance_isSamples, dtype)
        rv_mx = prepare_mxnet_array(rv, rv_isSamples, dtype)
        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape

        rbf = RBF(2, True, 1., 1., 'rbf', None, dtype)

        m = Model()
        m.X_var = Variable(shape=(5,2))
        m.Y = GaussianProcess.define_variable(X=m.X_var, kernel=rbf, shape=rv_shape, dtype=dtype)

        gp = m.clone()[0].Y.factor

        variables = {gp.X.uuid: X_mx, gp.rbf_lengthscale.uuid: rbf_lengthscale_mx, gp.rbf_variance.uuid: rbf_variance_mx, gp.random_variable.uuid: rv_mx}
        log_pdf_rt = gp.log_pdf(F=mx.nd, variables=variables).asnumpy()

        log_pdf_np = []
        for i in range(num_samples):
            X_i = X[i] if X_isSamples else X
            lengthscale_i = rbf_lengthscale[i] if rbf_lengthscale_isSamples else rbf_lengthscale
            variance_i = rbf_variance[i] if rbf_variance_isSamples else rbf_variance
            rv_i = rv[i] if rv_isSamples else rv
            rbf_np = GPy.kern.RBF(input_dim=2, ARD=True)
            rbf_np.lengthscale = lengthscale_i
            rbf_np.variance = variance_i
            K_np = rbf_np.K(X_i)
            log_pdf_np.append(multivariate_normal.logpdf(rv_i[:,0], mean=None, cov=K_np))
        log_pdf_np = np.array(log_pdf_np)
        isSamples_any = any([X_isSamples, rbf_lengthscale_isSamples, rbf_variance_isSamples, rv_isSamples])
        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == isSamples_any
        if isSamples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        assert np.allclose(log_pdf_np, log_pdf_rt)
