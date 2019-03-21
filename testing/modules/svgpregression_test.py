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
import warnings
import mxnet as mx
import mxnet.gluon.nn as nn
import numpy as np
from mxfusion.models import Model
from mxfusion.modules.gp_modules import SVGPRegression
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.components.distributions import Normal
from mxfusion.components import Variable
from mxfusion.components.functions import MXFusionGluonFunction
from mxfusion.inference import Inference, MAP, ModulePredictionAlgorithm, TransferInference, create_Gaussian_meanfield, StochasticVariationalInference, GradBasedInference, ForwardSamplingAlgorithm
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.modules.gp_modules.svgp_regression import SVGPRegressionSamplingPrediction


import matplotlib
matplotlib.use('Agg')
import GPy

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestSVGPRegressionModule(object):

    def gen_data(self):
        np.random.seed(0)
        D = 1
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, D)
        Z = np.random.rand(3, 3)
        qU_mean = np.random.rand(3, D)
        qU_cov_W = np.random.rand(3, 3)
        qU_cov_diag = np.random.rand(3,)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)
        variance = np.random.rand(1)
        qU_chol = np.linalg.cholesky(
            qU_cov_W.dot(qU_cov_W.T)+np.diag(qU_cov_diag))[None, :, :]
        return D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol

    def gen_mxfusion_model(self, dtype, D, Z, noise_var, lengthscale, variance,
                           rand_gen=None):
        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.Z = Variable(shape=(3, 3), initial_value=mx.nd.array(Z, dtype=dtype))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = SVGPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, inducing_inputs=m.Z, shape=(m.N, D), dtype=dtype)
        gp = m.Y.factor
        m.Y.factor.svgp_log_pdf.jitter = 1e-8
        return m, gp

    def gen_mxfusion_model_w_mean(self, dtype, D, Z, noise_var, lengthscale,
                                  variance, rand_gen=None):
        net = nn.HybridSequential(prefix='nn_')
        with net.name_scope():
            net.add(nn.Dense(D, flatten=False, activation="tanh",
                             in_units=3, dtype=dtype))
        net.initialize(mx.init.Xavier(magnitude=3))

        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.Z = Variable(shape=(3, 3), initial_value=mx.nd.array(Z, dtype=dtype))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.mean_func = MXFusionGluonFunction(net, num_outputs=1,
                                            broadcastable=True)
        m.Y = SVGPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, mean=m.mean_func(m.X), inducing_inputs=m.Z, shape=(m.N, D), dtype=dtype)
        gp = m.Y.factor
        return m, gp, net

    def test_log_pdf(self):
        D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol = self.gen_data()

        m_gpy = GPy.core.SVGP(X=X, Y=Y, Z=Z, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), likelihood=GPy.likelihoods.Gaussian(variance=noise_var))
        m_gpy.q_u_mean = qU_mean
        m_gpy.q_u_chol = GPy.util.choleskies.triang_to_flat(qU_chol)

        l_gpy = m_gpy.log_likelihood()

        dtype = 'float64'
        m, gp = self.gen_mxfusion_model(dtype, D, Z, noise_var, lengthscale,
                                        variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))
        l_mf = -loss

        assert np.allclose(l_mf.asnumpy(), l_gpy)

    def test_log_pdf_w_mean(self):
        D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol = self.gen_data()
        dtype = 'float64'
        m, gp, net = self.gen_mxfusion_model_w_mean(dtype, D, Z, noise_var,
                                                    lengthscale, variance)
        mean = net(mx.nd.array(X, dtype=dtype)).asnumpy()

        m_gpy = GPy.core.SVGP(X=X, Y=Y-mean, Z=Z, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), likelihood=GPy.likelihoods.Gaussian(variance=noise_var))
        m_gpy.q_u_mean = qU_mean
        m_gpy.q_u_chol = GPy.util.choleskies.triang_to_flat(qU_chol)

        l_gpy = m_gpy.log_likelihood()

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))
        l_mf = -loss

        assert np.allclose(l_mf.asnumpy(), l_gpy)

    def test_log_pdf_w_samples_of_noise_var(self):
        D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol = self.gen_data()
        dtype = 'float64'
        D = 2
        Y = np.random.rand(10, D)
        qU_mean = np.random.rand(3, D)

        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.Z = Variable(shape=(3, 3), initial_value=mx.nd.array(Z, dtype=dtype))
        m.noise_var = Variable(transformation=PositiveTransformation(), shape=(m.N, D))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = SVGPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, inducing_inputs=m.Z, shape=(m.N, D), dtype=dtype)
        gp = m.Y.factor
        m.Y.factor.svgp_log_pdf.jitter = 1e-8

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype), max_iter=1)


    def test_prediction(self):
        D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol = self.gen_data()
        Xt = np.random.rand(5, 3)

        m_gpy = GPy.core.SVGP(X=X, Y=Y, Z=Z, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), likelihood=GPy.likelihoods.Gaussian(variance=noise_var))
        m_gpy.q_u_mean = qU_mean
        m_gpy.q_u_chol = GPy.util.choleskies.triang_to_flat(qU_chol)

        dtype = 'float64'
        m, gp = self.gen_mxfusion_model(dtype, D, Z, noise_var, lengthscale,
                                        variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        # noise_free, diagonal
        mu_gpy, var_gpy = m_gpy.predict_noiseless(Xt)

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy, var_mf), (var_gpy, var_mf)
        assert var_gpy.shape == var_mf.shape

        # noisy, diagonal
        mu_gpy, var_gpy = m_gpy.predict(Xt)

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        infr2.inference_algorithm.model.Y.factor.svgp_predict.noise_free = False
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy, var_mf), (var_gpy, var_mf)
        assert var_gpy.shape == var_mf.shape

        m.Y.factor.svgp_predict.jitter = 1e-8

        # noise_free, full_cov
        mu_gpy, var_gpy = m_gpy.predict_noiseless(Xt, full_cov=True)

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        infr2.inference_algorithm.model.Y.factor.svgp_predict.diagonal_variance = False
        infr2.inference_algorithm.model.Y.factor.svgp_predict.noise_free = True
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy, var_mf), (var_gpy, var_mf)
        assert var_gpy.shape == var_mf.shape

        # noisy, full_cov
        mu_gpy, var_gpy = m_gpy.predict(Xt, full_cov=True)

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        infr2.inference_algorithm.model.Y.factor.svgp_predict.diagonal_variance = False
        infr2.inference_algorithm.model.Y.factor.svgp_predict.noise_free = False
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy, var_mf), (var_gpy, var_mf)
        assert var_gpy.shape == var_mf.shape

    def test_draw_samples(self):
        D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol = self.gen_data()
        dtype = 'float64'
        m, gp = self.gen_mxfusion_model(dtype, D, Z, noise_var, lengthscale,
                                        variance)

        observed = [m.X]
        infr = Inference(ForwardSamplingAlgorithm(
            m, observed, num_samples=2, target_variables=[m.Y]), dtype=dtype)
        samples = infr.run(X=mx.nd.array(X, dtype=dtype))[0]
        assert samples.shape == (2,) + Y.shape

    def test_prediction_w_mean(self):
        D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol = self.gen_data()
        Xt = np.random.rand(5, 3)
        dtype = 'float64'
        m, gp, net = self.gen_mxfusion_model_w_mean(dtype, D, Z, noise_var,
                                                    lengthscale, variance)
        mean = net(mx.nd.array(X, dtype=dtype)).asnumpy()
        mean_t = net(mx.nd.array(Xt, dtype=dtype)).asnumpy()

        m_gpy = GPy.core.SVGP(X=X, Y=Y-mean, Z=Z, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), likelihood=GPy.likelihoods.Gaussian(variance=noise_var))
        m_gpy.q_u_mean = qU_mean
        m_gpy.q_u_chol = GPy.util.choleskies.triang_to_flat(qU_chol)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        # noise_free, diagonal
        mu_gpy, var_gpy = m_gpy.predict_noiseless(Xt)
        mu_gpy += mean_t

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf, rtol=1e-04, atol=1e-05), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy, var_mf, rtol=1e-04, atol=1e-05), (var_gpy, var_mf)

    def test_sampling_prediction(self):
        D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol = self.gen_data()
        Xt = np.random.rand(5, 3)

        m_gpy = GPy.core.SVGP(X=X, Y=Y, Z=Z, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), likelihood=GPy.likelihoods.Gaussian(variance=noise_var))
        m_gpy.q_u_mean = qU_mean
        m_gpy.q_u_chol = GPy.util.choleskies.triang_to_flat(qU_chol)

        dtype = 'float64'
        m, gp = self.gen_mxfusion_model(dtype, D, Z, noise_var, lengthscale,
                                        variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        # noise_free, diagonal
        infr_pred = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y], num_samples=5),
                                      infr_params=infr.params)
        gp = m.Y.factor
        gp.attach_prediction_algorithms(
            targets=gp.output_names, conditionals=gp.input_names,
            algorithm=SVGPRegressionSamplingPrediction(
                gp._module_graph, gp._extra_graphs[0], [gp._module_graph.X]),
            alg_name='svgp_predict')
        gp.svgp_predict.diagonal_variance = False
        gp.svgp_predict.noise_free = False
        gp.svgp_predict.jitter = 1e-6

        y_samples = infr_pred.run(X=mx.nd.array(Xt, dtype=dtype))[0].asnumpy()

    def test_sampling_prediction_w_mean(self):
        D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol = self.gen_data()
        Xt = np.random.rand(20, 3)

        dtype = 'float64'
        m, gp, net = self.gen_mxfusion_model_w_mean(dtype, D, Z, noise_var,
                                                    lengthscale, variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        # noise_free, diagonal
        infr_pred = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y], num_samples=5),
                                      infr_params=infr.params)
        gp = m.Y.factor
        gp.attach_prediction_algorithms(
            targets=gp.output_names, conditionals=gp.input_names,
            algorithm=SVGPRegressionSamplingPrediction(
                gp._module_graph, gp._extra_graphs[0], [gp._module_graph.X]),
            alg_name='svgp_predict')
        gp.svgp_predict.diagonal_variance = True
        gp.svgp_predict.noise_free = False
        gp.svgp_predict.jitter = 1e-6

        y_samples = infr_pred.run(X=mx.nd.array(Xt, dtype=dtype))[0].asnumpy()

    def test_with_samples(self):
        from mxfusion.common import config
        config.DEFAULT_DTYPE = 'float64'
        dtype = 'float64'

        D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol = self.gen_data()

        m = Model()
        m.N = Variable()
        m.X = Normal.define_variable(mean=0, variance=1, shape=(m.N, 3))
        m.Z = Variable(shape=(3, 3), initial_value=mx.nd.array(Z, dtype=dtype))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = SVGPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, inducing_inputs=m.Z, shape=(m.N, D), dtype=dtype)
        gp = m.Y.factor
        gp.svgp_log_pdf.jitter = 1e-8

        q = create_Gaussian_meanfield(model=m, observed=[m.Y])

        infr = GradBasedInference(
            inference_algorithm=StochasticVariationalInference(
                model=m, posterior=q, num_samples=10, observed=[m.Y]))
        infr.initialize(Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)
        infr.run(Y=mx.nd.array(Y, dtype='float64'), max_iter=2,
                 learning_rate=0.1, verbose=True)

        infr2 = Inference(ForwardSamplingAlgorithm(
            model=m, observed=[m.X], num_samples=5))
        infr2.run(X=mx.nd.array(X, dtype='float64'))

        infr_pred = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params)
        xt = np.random.rand(13, 3)
        res = infr_pred.run(X=mx.nd.array(xt, dtype=dtype))[0]

        gp = m.Y.factor
        gp.attach_prediction_algorithms(
            targets=gp.output_names, conditionals=gp.input_names,
            algorithm=SVGPRegressionSamplingPrediction(
                gp._module_graph, gp._extra_graphs[0], [gp._module_graph.X]),
            alg_name='svgp_predict')
        gp.svgp_predict.diagonal_variance = False
        gp.svgp_predict.jitter = 1e-6

        infr_pred2 = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params)
        xt = np.random.rand(13, 3)
        res = infr_pred2.run(X=mx.nd.array(xt, dtype=dtype))[0]

    def test_module_clone(self):
        D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag, qU_chol = self.gen_data()
        dtype = 'float64'

        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = SVGPRegression.define_variable(X=mx.nd.zeros((2, 3)), kernel=kernel, noise_var=mx.nd.ones((1,)), dtype=dtype)
        m.clone()
