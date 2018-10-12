import pytest
import mxnet as mx
import numpy as np
from mxfusion.models import Model
from mxfusion.modules.gp_modules import SparseGPRegression
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.components import Variable
from mxfusion.inference import Inference, MAP
from mxfusion.components.variables.var_trans import PositiveTransformation

import matplotlib
matplotlib.use('Agg')
import GPy


class TestSparseGPRegressionModule(object):

    def test_log_pdf(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, 1)
        Z = np.random.rand(3, 3)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)
        variance = np.random.rand(1)

        m_gpy = GPy.models.SparseGPRegression(X=X, Y=Y, Z=Z, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), num_inducing=3)
        m_gpy.likelihood.variance = noise_var

        l_gpy = m_gpy.log_likelihood()

        dtype = 'float64'
        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.Z = Variable(shape=(3, 3), initial_value=mx.nd.array(Z, dtype=dtype))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = SparseGPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, inducing_inputs=m.Z, shape=(m.N, 1), dtype=dtype)
        m.Y.factor.jitter = 1e-8

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))
        l_mf = -loss

        print(l_mf.asnumpy(), l_gpy)
        assert False
        assert np.allclose(l_mf.asnumpy(), l_gpy)
