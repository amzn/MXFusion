import pytest
import mxnet as mx
import numpy as np
from mxfusion.models import Model
from mxfusion.components.variables.runtime_variable import array_has_samples, get_num_samples
from mxfusion.components.distributions import ConditionalGaussianProcess
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.components.variables import Variable
from mxfusion.util.testutils import prepare_mxnet_array
from mxfusion.util.testutils import MockMXNetRandomGenerator
from scipy.stats import multivariate_normal
import matplotlib
matplotlib.use('Agg')
import GPy


@pytest.mark.usefixtures("set_seed")
class TestConditionalGaussianProcessDistribution(object):

    @pytest.mark.parametrize("dtype, X, X_isSamples, X_cond, X_cond_isSamples, Y_cond, Y_cond_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, rv, rv_isSamples, num_samples", [
        (np.float64, np.random.rand(5,2), False, np.random.rand(8,2), False, np.random.rand(8,1), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, np.random.rand(3,5,1), True, 3),
        (np.float64, np.random.rand(3,5,2), True, np.random.rand(8,2), False, np.random.rand(8,1), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, np.random.rand(5,1), False, 3),
        (np.float64, np.random.rand(3,5,2), True, np.random.rand(8,2), False, np.random.rand(3,8,1), True, np.random.rand(3,2)+0.1, True, np.random.rand(3,1)+0.1, True, np.random.rand(3,5,1), True, 3),
        (np.float64, np.random.rand(5,2), False, np.random.rand(8,2), False, np.random.rand(8,1), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, np.random.rand(5,1), False, 1),
        ])
    def test_log_pdf(self, dtype, X, X_isSamples, X_cond, X_cond_isSamples, Y_cond, Y_cond_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples,
                        rv, rv_isSamples, num_samples):
        from scipy.linalg.lapack import dtrtrs
        X_mx = prepare_mxnet_array(X, X_isSamples, dtype)
        X_cond_mx = prepare_mxnet_array(X_cond, X_cond_isSamples, dtype)
        Y_cond_mx = prepare_mxnet_array(Y_cond, Y_cond_isSamples, dtype)
        rbf_lengthscale_mx = prepare_mxnet_array(rbf_lengthscale, rbf_lengthscale_isSamples, dtype)
        rbf_variance_mx = prepare_mxnet_array(rbf_variance, rbf_variance_isSamples, dtype)
        rv_mx = prepare_mxnet_array(rv, rv_isSamples, dtype)
        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape

        rbf = RBF(2, True, 1., 1., 'rbf', None, dtype)
        X_var = Variable(shape=(5,2))
        X_cond_var = Variable(shape=(8,2))
        Y_cond_var = Variable(shape=(8,1))
        gp = ConditionalGaussianProcess.define_variable(X=X_var, X_cond=X_cond_var, Y_cond=Y_cond_var, kernel=rbf, shape=rv_shape, dtype=dtype).factor

        variables = {gp.X.uuid: X_mx, gp.X_cond.uuid: X_cond_mx, gp.Y_cond.uuid: Y_cond_mx, gp.rbf_lengthscale.uuid: rbf_lengthscale_mx, gp.rbf_variance.uuid: rbf_variance_mx, gp.random_variable.uuid: rv_mx}
        log_pdf_rt = gp.log_pdf(F=mx.nd, variables=variables).asnumpy()

        log_pdf_np = []
        for i in range(num_samples):
            X_i = X[i] if X_isSamples else X
            X_cond_i = X_cond[i] if X_cond_isSamples else X_cond
            Y_cond_i = Y_cond[i] if Y_cond_isSamples else Y_cond
            lengthscale_i = rbf_lengthscale[i] if rbf_lengthscale_isSamples else rbf_lengthscale
            variance_i = rbf_variance[i] if rbf_variance_isSamples else rbf_variance
            rv_i = rv[i] if rv_isSamples else rv
            rbf_np = GPy.kern.RBF(input_dim=2, ARD=True)
            rbf_np.lengthscale = lengthscale_i
            rbf_np.variance = variance_i
            K_np = rbf_np.K(X_i)
            Kc_np = rbf_np.K(X_cond_i, X_i)
            Kcc_np = rbf_np.K(X_cond_i)

            L = np.linalg.cholesky(Kcc_np)
            LInvY = dtrtrs(L, Y_cond_i, lower=1, trans=0)[0]
            LinvKxt = dtrtrs(L, Kc_np, lower=1, trans=0)[0]

            mu = LinvKxt.T.dot(LInvY)
            cov = K_np - LinvKxt.T.dot(LinvKxt)
            log_pdf_np.append(multivariate_normal.logpdf(rv_i[:,0], mean=mu[:,0], cov=cov))
        log_pdf_np = np.array(log_pdf_np)
        isSamples_any = any([X_isSamples, rbf_lengthscale_isSamples, rbf_variance_isSamples, rv_isSamples])
        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == isSamples_any
        if isSamples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        assert np.allclose(log_pdf_np, log_pdf_rt)

    @pytest.mark.parametrize("dtype, X, X_isSamples, X_cond, X_cond_isSamples, Y_cond, Y_cond_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, rv_shape, num_samples", [
        (np.float64, np.random.rand(5,2), False, np.random.rand(8,2), False, np.random.rand(8,1), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, (5,1), 3),
        (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,8,2), True, np.random.rand(8,1), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, (5,1), 3),
        (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,8,2), True, np.random.rand(3,8,1), True, np.random.rand(3,2)+0.1, True, np.random.rand(3,1)+0.1, True, (5,1), 3),
        (np.float64, np.random.rand(5,2), False, np.random.rand(8,2), False, np.random.rand(8,1), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, (5,1), 1),
        ])
    def test_draw_samples(self, dtype, X, X_isSamples, X_cond, X_cond_isSamples, Y_cond, Y_cond_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples,
                        rv_shape, num_samples):
        from scipy.linalg.lapack import dtrtrs
        X_mx = prepare_mxnet_array(X, X_isSamples, dtype)
        X_cond_mx = prepare_mxnet_array(X_cond, X_cond_isSamples, dtype)
        Y_cond_mx = prepare_mxnet_array(Y_cond, Y_cond_isSamples, dtype)
        rbf_lengthscale_mx = prepare_mxnet_array(rbf_lengthscale, rbf_lengthscale_isSamples, dtype)
        rbf_variance_mx = prepare_mxnet_array(rbf_variance, rbf_variance_isSamples, dtype)

        rand = np.random.randn(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        rbf = RBF(2, True, 1., 1., 'rbf', None, dtype)
        X_var = Variable(shape=(5,2))
        X_cond_var = Variable(shape=(8,2))
        Y_cond_var = Variable(shape=(8,1))
        gp = ConditionalGaussianProcess.define_variable(X=X_var, X_cond=X_cond_var, Y_cond=Y_cond_var, kernel=rbf, shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor

        variables = {gp.X.uuid: X_mx, gp.X_cond.uuid: X_cond_mx, gp.Y_cond.uuid: Y_cond_mx, gp.rbf_lengthscale.uuid: rbf_lengthscale_mx, gp.rbf_variance.uuid: rbf_variance_mx}
        samples_rt = gp.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples).asnumpy()

        samples_np = []
        for i in range(num_samples):
            X_i = X[i] if X_isSamples else X
            X_cond_i = X_cond[i] if X_cond_isSamples else X_cond
            Y_cond_i = Y_cond[i] if Y_cond_isSamples else Y_cond
            lengthscale_i = rbf_lengthscale[i] if rbf_lengthscale_isSamples else rbf_lengthscale
            variance_i = rbf_variance[i] if rbf_variance_isSamples else rbf_variance
            rand_i = rand[i]
            rbf_np = GPy.kern.RBF(input_dim=2, ARD=True)
            rbf_np.lengthscale = lengthscale_i
            rbf_np.variance = variance_i
            K_np = rbf_np.K(X_i)
            Kc_np = rbf_np.K(X_cond_i, X_i)
            Kcc_np = rbf_np.K(X_cond_i)

            L = np.linalg.cholesky(Kcc_np)
            LInvY = dtrtrs(L, Y_cond_i, lower=1, trans=0)[0]
            LinvKxt = dtrtrs(L, Kc_np, lower=1, trans=0)[0]

            mu = LinvKxt.T.dot(LInvY)
            cov = K_np - LinvKxt.T.dot(LinvKxt)
            L_cov_np = np.linalg.cholesky(cov)
            sample_np = mu + L_cov_np.dot(rand_i)
            samples_np.append(sample_np)
        samples_np = np.array(samples_np)
        assert np.issubdtype(samples_rt.dtype, dtype)
        assert get_num_samples(mx.nd, samples_rt) == num_samples
        print(samples_np, samples_rt)
        assert np.allclose(samples_np, samples_rt)

    @pytest.mark.parametrize("dtype, X, X_isSamples, X_cond, X_cond_isSamples, Y_cond, Y_cond_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, rv, rv_isSamples, num_samples", [
        (np.float64, np.random.rand(5,2), False, np.random.rand(8,2), False, np.random.rand(8,1), False, np.random.rand(2)+0.1, False, np.random.rand(1)+0.1, False, np.random.rand(3,5,1), True, 3),
        ])
    def test_clone_cond_gp(self, dtype, X, X_isSamples, X_cond, X_cond_isSamples, Y_cond, Y_cond_isSamples, rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples,
                        rv, rv_isSamples, num_samples):
        from scipy.linalg.lapack import dtrtrs
        X_mx = prepare_mxnet_array(X, X_isSamples, dtype)
        X_cond_mx = prepare_mxnet_array(X_cond, X_cond_isSamples, dtype)
        Y_cond_mx = prepare_mxnet_array(Y_cond, Y_cond_isSamples, dtype)
        rbf_lengthscale_mx = prepare_mxnet_array(rbf_lengthscale, rbf_lengthscale_isSamples, dtype)
        rbf_variance_mx = prepare_mxnet_array(rbf_variance, rbf_variance_isSamples, dtype)
        rv_mx = prepare_mxnet_array(rv, rv_isSamples, dtype)
        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape

        rbf = RBF(2, True, 1., 1., 'rbf', None, dtype)
        m = Model()
        m.X_var = Variable(shape=(5,2))
        m.X_cond_var = Variable(shape=(8,2))
        m.Y_cond_var = Variable(shape=(8,1))
        m.Y = ConditionalGaussianProcess.define_variable(X=m.X_var, X_cond=m.X_cond_var, Y_cond=m.Y_cond_var, kernel=rbf, shape=rv_shape, dtype=dtype)

        gp = m.clone()[0].Y.factor

        variables = {gp.X.uuid: X_mx, gp.X_cond.uuid: X_cond_mx, gp.Y_cond.uuid: Y_cond_mx, gp.rbf_lengthscale.uuid: rbf_lengthscale_mx, gp.rbf_variance.uuid: rbf_variance_mx, gp.random_variable.uuid: rv_mx}
        log_pdf_rt = gp.log_pdf(F=mx.nd, variables=variables).asnumpy()

        log_pdf_np = []
        for i in range(num_samples):
            X_i = X[i] if X_isSamples else X
            X_cond_i = X_cond[i] if X_cond_isSamples else X_cond
            Y_cond_i = Y_cond[i] if Y_cond_isSamples else Y_cond
            lengthscale_i = rbf_lengthscale[i] if rbf_lengthscale_isSamples else rbf_lengthscale
            variance_i = rbf_variance[i] if rbf_variance_isSamples else rbf_variance
            rv_i = rv[i] if rv_isSamples else rv
            rbf_np = GPy.kern.RBF(input_dim=2, ARD=True)
            rbf_np.lengthscale = lengthscale_i
            rbf_np.variance = variance_i
            K_np = rbf_np.K(X_i)
            Kc_np = rbf_np.K(X_cond_i, X_i)
            Kcc_np = rbf_np.K(X_cond_i)

            L = np.linalg.cholesky(Kcc_np)
            LInvY = dtrtrs(L, Y_cond_i, lower=1, trans=0)[0]
            LinvKxt = dtrtrs(L, Kc_np, lower=1, trans=0)[0]

            mu = LinvKxt.T.dot(LInvY)
            cov = K_np - LinvKxt.T.dot(LinvKxt)
            log_pdf_np.append(multivariate_normal.logpdf(rv_i[:,0], mean=mu[:,0], cov=cov))
        log_pdf_np = np.array(log_pdf_np)
        isSamples_any = any([X_isSamples, rbf_lengthscale_isSamples, rbf_variance_isSamples, rv_isSamples])
        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == isSamples_any
        if isSamples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        assert np.allclose(log_pdf_np, log_pdf_rt)
