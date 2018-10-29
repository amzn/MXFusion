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
from mxfusion.models import Model
from mxfusion.modules.gp_modules import SparseGPRegression
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.components import Variable
from mxfusion.inference import Inference, MAP, ModulePredictionAlgorithm, TransferInference
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.modules.gp_modules.sparsegp_regression import SparseGPRegressionSamplingPrediction


import matplotlib
matplotlib.use('Agg')
import GPy


class TestSparseGPRegressionModule(object):

    def test_log_pdf(self):
        np.random.seed(0)
        D = 2
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, D)
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
        m.Y = SparseGPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, inducing_inputs=m.Z, shape=(m.N, D), dtype=dtype)
        m.Y.factor.sgp_log_pdf.jitter = 1e-8

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))
        l_mf = -loss

        assert np.allclose(l_mf.asnumpy(), l_gpy)

    def test_prediction(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, 1)
        Z = np.random.rand(3, 3)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)/10.
        variance = np.random.rand(1)
        Xt = np.random.rand(20, 3)

        m_gpy = GPy.models.SparseGPRegression(X=X, Y=Y, Z=Z, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), num_inducing=3)
        m_gpy.likelihood.variance = noise_var

        dtype = 'float64'
        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.Z = Variable(shape=(3, 3), initial_value=mx.nd.array(Z, dtype=dtype))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = SparseGPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, inducing_inputs=m.Z, shape=(m.N, 1), dtype=dtype)
        m.Y.factor.sgp_log_pdf.jitter = 1e-8

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        # noise_free, diagonal
        mu_gpy, var_gpy = m_gpy.predict_noiseless(Xt)

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy[:,0], var_mf), (var_gpy[:,0], var_mf)

        # noisy, diagonal
        mu_gpy, var_gpy = m_gpy.predict(Xt)

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        infr2.inference_algorithm.model.Y.factor.sgp_predict.noise_free = False
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy[:,0], var_mf), (var_gpy[:,0], var_mf)

        # noise_free, full_cov
        mu_gpy, var_gpy = m_gpy.predict_noiseless(Xt, full_cov=True)

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        infr2.inference_algorithm.model.Y.factor.sgp_predict.diagonal_variance = False
        infr2.inference_algorithm.model.Y.factor.sgp_predict.noise_free = True
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy, var_mf), (var_gpy, var_mf)

        # noisy, full_cov
        mu_gpy, var_gpy = m_gpy.predict(Xt, full_cov=True)

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        infr2.inference_algorithm.model.Y.factor.sgp_predict.diagonal_variance = False
        infr2.inference_algorithm.model.Y.factor.sgp_predict.noise_free = False
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy, var_mf), (var_gpy, var_mf)

    def test_sampling_prediction(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, 1)
        Z = np.random.rand(3, 3)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)/10.
        variance = np.random.rand(1)
        Xt = np.random.rand(20, 3)

        m_gpy = GPy.models.SparseGPRegression(X=X, Y=Y, Z=Z, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), num_inducing=3)
        m_gpy.likelihood.variance = noise_var

        dtype = 'float64'
        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.Z = Variable(shape=(3, 3), initial_value=mx.nd.array(Z, dtype=dtype))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = SparseGPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, inducing_inputs=m.Z, shape=(m.N, 1), dtype=dtype)
        m.Y.factor.sgp_log_pdf.jitter = 1e-8

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        # noise_free, diagonal

        infr_pred = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y], num_samples=5),
                                      infr_params=infr.params)
        gp = m.Y.factor
        gp.attach_prediction_algorithms(
            targets=gp.output_names, conditionals=gp.input_names,
            algorithm=SparseGPRegressionSamplingPrediction(
                gp._module_graph, gp._extra_graphs[0], [gp._module_graph.X]),
            alg_name='sgp_predict')
        gp.sgp_predict.diagonal_variance = False
        gp.sgp_predict.jitter = 1e-6

        y_samples = infr_pred.run(X=mx.nd.array(Xt, dtype=dtype))[0].asnumpy()

        # TODO: Check the correctness of the sampling
