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
from mxfusion.modules.gp_modules import GPRegression
from mxfusion.components.distributions.gp.kernels import RBF, White
from mxfusion.components.distributions import GaussianProcess
from mxfusion.components import Variable
from mxfusion.inference import Inference, MAP, ModulePredictionAlgorithm, TransferInference
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.inference.forward_sampling import ForwardSamplingAlgorithm
from mxfusion.util.testutils import MockMXNetRandomGenerator
from mxfusion.modules.gp_modules.gp_regression import GPRegressionSamplingPrediction

import matplotlib
matplotlib.use('Agg')
import GPy


class TestGPRegressionModule(object):

    def test_log_pdf(self):
        np.random.seed(0)
        D = 2
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, D)
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
        m.Y = GPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, shape=(m.N, D), dtype=dtype)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))
        l_mf = -loss

        assert np.allclose(l_mf.asnumpy(), l_gpy)

    def test_draw_samples(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, 1)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)
        variance = np.random.rand(1)
        dtype = 'float64'

        rand_gen = MockMXNetRandomGenerator(mx.nd.array(np.random.rand(20), dtype=dtype))

        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = GPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, shape=(m.N, 1), dtype=dtype, rand_gen=rand_gen)

        observed = [m.X]
        infr = Inference(ForwardSamplingAlgorithm(
            m, observed, num_samples=2, target_variables=[m.Y]), dtype=dtype)

        samples = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))[0].asnumpy()

        kern = RBF(3, True, name='rbf', dtype=dtype) + White(3, dtype=dtype)
        X_var = Variable(shape=(10, 3))
        gp = GaussianProcess.define_variable(X=X_var, kernel=kern, shape=(10, 1), dtype=dtype, rand_gen=rand_gen).factor

        variables = {gp.X.uuid: mx.nd.expand_dims(mx.nd.array(X, dtype=dtype), axis=0), gp.add_rbf_lengthscale.uuid: mx.nd.expand_dims(mx.nd.array(lengthscale, dtype=dtype), axis=0), gp.add_rbf_variance.uuid: mx.nd.expand_dims(mx.nd.array(variance, dtype=dtype), axis=0), gp.add_white_variance.uuid: mx.nd.expand_dims(mx.nd.array(noise_var, dtype=dtype), axis=0)}
        samples_2 = gp.draw_samples(F=mx.nd, variables=variables, num_samples=2).asnumpy()

        assert np.allclose(samples, samples_2), (samples, samples_2)

    def test_prediction(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        Xt = np.random.rand(20, 3)
        Y = np.random.rand(10, 1)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)
        variance = np.random.rand(1)

        m_gpy = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), noise_var=noise_var)

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
        infr2.inference_algorithm.model.Y.factor.gp_predict.noise_free = False
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy[:,0], var_mf), (var_gpy[:,0], var_mf)

        # noise_free, full_cov
        mu_gpy, var_gpy = m_gpy.predict_noiseless(Xt, full_cov=True)

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        infr2.inference_algorithm.model.Y.factor.gp_predict.diagonal_variance = False
        infr2.inference_algorithm.model.Y.factor.gp_predict.noise_free = True
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy, var_mf), (var_gpy, var_mf)

        # noisy, full_cov
        mu_gpy, var_gpy = m_gpy.predict(Xt, full_cov=True)

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        infr2.inference_algorithm.model.Y.factor.gp_predict.diagonal_variance = False
        infr2.inference_algorithm.model.Y.factor.gp_predict.noise_free = False
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]
        print((var_gpy, var_mf))

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy, var_mf), (var_gpy, var_mf)

    def test_sampling_prediction(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        Xt = np.random.rand(20, 3)
        Y = np.random.rand(10, 1)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)
        variance = np.random.rand(1)

        m_gpy = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), noise_var=noise_var)

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

        infr_pred = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y], num_samples=5),
                                      infr_params=infr.params)
        gp = m.Y.factor
        gp.attach_prediction_algorithms(
            targets=gp.output_names, conditionals=gp.input_names,
            algorithm=GPRegressionSamplingPrediction(
                gp._module_graph, gp._extra_graphs[0], [gp._module_graph.X]),
            alg_name='gp_predict')
        gp.gp_predict.diagonal_variance = False
        gp.gp_predict.jitter = 1e-6

        y_samples = infr_pred.run(X=mx.nd.array(Xt, dtype=dtype))[0].asnumpy()

        # TODO: Check the correctness of the sampling
