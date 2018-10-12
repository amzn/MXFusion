import pytest
import mxnet as mx
import numpy as np
from mxfusion.models import Model
from mxfusion.modules.gp_modules import GPRegression
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.components import Variable
from mxfusion.inference import Inference, MAP
from mxfusion.components.variables.var_trans import PositiveTransformation

import matplotlib
matplotlib.use('Agg')
import GPy


class TestGaussianProcessDistribution(object):

    def test_log_pdf(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, 1)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)
        variance = np.random.rand(1)

        m_gpy = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), noise_var=noise_var)

        l_gpy = m_gpy.log_likelihood()

        dtype = 'float64'
        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = GPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, shape=(m.N, 1), dtype=dtype)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))
        l_mf = -loss

        assert np.allclose(l_mf.asnumpy(), l_gpy)
