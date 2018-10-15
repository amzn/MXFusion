import pytest
import mxnet as mx
import numpy as np
from mxfusion.models import Model
from mxfusion.modules.gp_modules import SVGPRegression
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.components import Variable
from mxfusion.inference import Inference, MAP
from mxfusion.components.variables.var_trans import PositiveTransformation


import matplotlib
matplotlib.use('Agg')
import GPy


class TestSVGPRegressionModule(object):

    def test_log_pdf(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, 1)
        Z = np.random.rand(3, 3)
        qU_mean = np.random.rand(3, 1)
        qU_cov_W = np.random.rand(3, 3)
        qU_cov_diag = np.random.rand(3,)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)
        variance = np.random.rand(1)
        qU_chol = np.linalg.cholesky(qU_cov_W.dot(qU_cov_W.T)+np.diag(qU_cov_diag))[None,:,:]

        print(qU_chol[0].dot(qU_chol[0].T), qU_cov_W.dot(qU_cov_W.T)+np.diag(qU_cov_diag))

        m_gpy = GPy.core.SVGP(X=X, Y=Y, Z=Z, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), likelihood=GPy.likelihoods.Gaussian(variance=noise_var))
        m_gpy.q_u_mean = qU_mean
        m_gpy.q_u_chol = GPy.util.choleskies.triang_to_flat(qU_chol)

        l_gpy = m_gpy.log_likelihood()

        dtype = 'float64'
        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.Z = Variable(shape=(3, 3), initial_value=mx.nd.array(Z, dtype=dtype))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = SVGPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, inducing_inputs=m.Z, shape=(m.N, 1), dtype=dtype)
        gp = m.Y.factor

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))
        l_mf = -loss

        assert np.allclose(l_mf.asnumpy(), l_gpy)
