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


import warnings
import mxnet as mx
import mxnet.gluon.nn as nn
import numpy as np
from mxfusion.models import Model
from mxfusion.modules.gp_modules import GPRegression
from mxfusion.components.distributions.gp.kernels import RBF, White
from mxfusion.components.distributions import GaussianProcess, Normal
from mxfusion.components import Variable
from mxfusion.components.functions import MXFusionGluonFunction
from mxfusion.inference import Inference, MAP, TransferInference, create_Gaussian_meanfield, StochasticVariationalInference, GradBasedInference, ForwardSamplingAlgorithm, ModulePredictionAlgorithm
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.util.testutils import MockMXNetRandomGenerator
from mxfusion.modules.gp_modules.gp_regression import GPRegressionSamplingPrediction

import matplotlib
matplotlib.use('Agg')
import GPy

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestGPRegressionModule(object):

    def gen_data(self):
        np.random.seed(0)
        D = 2
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, D)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)
        variance = np.random.rand(1)
        return D, X, Y, noise_var, lengthscale, variance

    def gen_mxfusion_model(self, dtype, D, noise_var, lengthscale, variance,
                           rand_gen=None):
        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = GPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, shape=(m.N, D), dtype=dtype, rand_gen=rand_gen)
        return m

    def gen_mxfusion_model_w_mean(self, dtype, D, noise_var, lengthscale,
                                  variance, rand_gen=None):
        net = nn.HybridSequential(prefix='nn_')
        with net.name_scope():
            net.add(nn.Dense(D, flatten=False, activation="tanh",
                             in_units=3, dtype=dtype))
        net.initialize(mx.init.Xavier(magnitude=3))

        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.mean_func = MXFusionGluonFunction(net, num_outputs=1,
                                            broadcastable=True)
        m.Y = GPRegression.define_variable(X=m.X, kernel=kernel, mean=m.mean_func(m.X), noise_var=m.noise_var, shape=(m.N, D), dtype=dtype, rand_gen=rand_gen)
        m.Y.factor.gp_log_pdf.jitter = 1e-6
        return m, net

    def test_log_pdf(self):
        D, X, Y, noise_var, lengthscale, variance = self.gen_data()

        # GPy log-likelihood
        m_gpy = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), noise_var=noise_var)
        l_gpy = m_gpy.log_likelihood()

        # MXFusion log-likelihood
        dtype = 'float64'
        m = self.gen_mxfusion_model(dtype, D, noise_var, lengthscale, variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))
        l_mf = -loss

        assert np.allclose(l_mf.asnumpy(), l_gpy)

    def test_log_pdf_w_mean(self):
        D, X, Y, noise_var, lengthscale, variance = self.gen_data()

        # MXFusion log-likelihood
        dtype = 'float64'
        m, net = self.gen_mxfusion_model_w_mean(
            dtype, D, noise_var, lengthscale, variance)

        mean = net(mx.nd.array(X, dtype=dtype)).asnumpy()

        # GPy log-likelihood
        m_gpy = GPy.models.GPRegression(X=X, Y=Y-mean, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), noise_var=noise_var)
        l_gpy = m_gpy.log_likelihood()

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))
        l_mf = -loss

        assert np.allclose(l_mf.asnumpy(), l_gpy)

    def test_draw_samples(self):
        D, X, Y, noise_var, lengthscale, variance = self.gen_data()
        dtype = 'float64'

        rand_gen = MockMXNetRandomGenerator(mx.nd.array(np.random.rand(20*D), dtype=dtype))

        m = self.gen_mxfusion_model(dtype, D, noise_var, lengthscale, variance, rand_gen)

        observed = [m.X]
        infr = Inference(ForwardSamplingAlgorithm(
            m, observed, num_samples=2, target_variables=[m.Y]), dtype=dtype)

        samples = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))[0].asnumpy()

        kern = RBF(3, True, name='rbf', dtype=dtype) + White(3, dtype=dtype)
        X_var = Variable(shape=(10, 3))
        gp = GaussianProcess.define_variable(X=X_var, kernel=kern, shape=(10, D), dtype=dtype, rand_gen=rand_gen).factor

        variables = {gp.X.uuid: mx.nd.expand_dims(mx.nd.array(X, dtype=dtype), axis=0), gp.add_rbf_lengthscale.uuid: mx.nd.expand_dims(mx.nd.array(lengthscale, dtype=dtype), axis=0), gp.add_rbf_variance.uuid: mx.nd.expand_dims(mx.nd.array(variance, dtype=dtype), axis=0), gp.add_white_variance.uuid: mx.nd.expand_dims(mx.nd.array(noise_var, dtype=dtype), axis=0)}
        samples_2 = gp.draw_samples(F=mx.nd, variables=variables, num_samples=2).asnumpy()

        assert np.allclose(samples, samples_2), (samples, samples_2)

    def test_draw_samples_w_mean(self):
        D, X, Y, noise_var, lengthscale, variance = self.gen_data()
        dtype = 'float64'

        rand_gen = MockMXNetRandomGenerator(mx.nd.array(np.random.rand(20*D), dtype=dtype))

        m, net = self.gen_mxfusion_model_w_mean(dtype, D, noise_var, lengthscale, variance, rand_gen)

        observed = [m.X]
        infr = Inference(ForwardSamplingAlgorithm(
            m, observed, num_samples=2, target_variables=[m.Y]), dtype=dtype)

        samples = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))[0].asnumpy()

        kern = RBF(3, True, name='rbf', dtype=dtype) + White(3, dtype=dtype)
        X_var = Variable(shape=(10, 3))
        mean_func = MXFusionGluonFunction(net, num_outputs=1,
                                          broadcastable=True)
        mean_var = mean_func(X_var)
        gp = GaussianProcess.define_variable(X=X_var, kernel=kern, mean=mean_var, shape=(10, D), dtype=dtype, rand_gen=rand_gen).factor

        variables = {gp.X.uuid: mx.nd.expand_dims(mx.nd.array(X, dtype=dtype), axis=0), gp.add_rbf_lengthscale.uuid: mx.nd.expand_dims(mx.nd.array(lengthscale, dtype=dtype), axis=0), gp.add_rbf_variance.uuid: mx.nd.expand_dims(mx.nd.array(variance, dtype=dtype), axis=0), gp.add_white_variance.uuid: mx.nd.expand_dims(mx.nd.array(noise_var, dtype=dtype), axis=0), mean_var.uuid: mx.nd.expand_dims(net(mx.nd.array(X, dtype=dtype)), axis=0)}
        samples_2 = gp.draw_samples(F=mx.nd, variables=variables, num_samples=2).asnumpy()

        assert np.allclose(samples, samples_2), (samples, samples_2)

    def test_prediction(self):
        D, X, Y, noise_var, lengthscale, variance = self.gen_data()
        Xt = np.random.rand(20, 3)

        m_gpy = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), noise_var=noise_var)

        dtype = 'float64'
        m = self.gen_mxfusion_model(dtype, D, noise_var, lengthscale, variance)

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

        assert np.allclose(mu_gpy, mu_mf), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy, var_mf), (var_gpy, var_mf)

    def test_prediction_w_mean(self):
        D, X, Y, noise_var, lengthscale, variance = self.gen_data()
        Xt = np.random.rand(20, 3)
        dtype = 'float64'

        m, net = self.gen_mxfusion_model_w_mean(
            dtype, D, noise_var, lengthscale, variance)

        mean = net(mx.nd.array(X, dtype=dtype)).asnumpy()
        mean_t = net(mx.nd.array(Xt, dtype=dtype)).asnumpy()

        m_gpy = GPy.models.GPRegression(X=X, Y=Y-mean, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), noise_var=noise_var)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        # noise_free, diagonal
        mu_gpy, var_gpy = m_gpy.predict_noiseless(Xt)
        mu_gpy += mean_t

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert np.allclose(mu_gpy, mu_mf, rtol=1e-04, atol=1e-05), (mu_gpy, mu_mf)
        assert np.allclose(var_gpy[:,0], var_mf, rtol=1e-04, atol=1e-05), (var_gpy[:,0], var_mf)

    def test_sampling_prediction(self):
        D, X, Y, noise_var, lengthscale, variance = self.gen_data()
        Xt = np.random.rand(20, 3)

        dtype = 'float64'
        m = self.gen_mxfusion_model(dtype, D, noise_var, lengthscale, variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype), max_iter=1)

        infr_pred = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y], num_samples=5),
                                      infr_params=infr.params)
        gp = m.Y.factor
        gp.attach_prediction_algorithms(
            targets=gp.output_names, conditionals=gp.input_names,
            algorithm=GPRegressionSamplingPrediction(
                gp._module_graph, gp._extra_graphs[0], [gp._module_graph.X]),
            alg_name='gp_predict')
        gp.gp_predict.diagonal_variance = False
        gp.gp_predict.noise_free = False
        gp.gp_predict.jitter = 1e-6

        y_samples = infr_pred.run(X=mx.nd.array(Xt, dtype=dtype))[0].asnumpy()
        # TODO: Check the correctness of the sampling

    def test_sampling_prediction_w_mean(self):
        D, X, Y, noise_var, lengthscale, variance = self.gen_data()
        Xt = np.random.rand(20, 3)
        dtype = 'float64'
        m, net = self.gen_mxfusion_model_w_mean(
            dtype, D, noise_var, lengthscale, variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype), max_iter=1)

        infr_pred = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y], num_samples=5),
                                      infr_params=infr.params)
        gp = m.Y.factor
        gp.attach_prediction_algorithms(
            targets=gp.output_names, conditionals=gp.input_names,
            algorithm=GPRegressionSamplingPrediction(
                gp._module_graph, gp._extra_graphs[0], [gp._module_graph.X]),
            alg_name='gp_predict')
        gp.gp_predict.diagonal_variance = True
        gp.gp_predict.noise_free = False
        gp.gp_predict.jitter = 1e-6

        y_samples = infr_pred.run(X=mx.nd.array(Xt, dtype=dtype))[0].asnumpy()

    def test_with_samples(self):
        from mxfusion.common import config
        config.DEFAULT_DTYPE = 'float64'
        dtype = 'float64'

        D, X, Y, noise_var, lengthscale, variance = self.gen_data()

        m = Model()
        m.N = Variable()
        m.X = Normal.define_variable(mean=0, variance=1, shape=(m.N, 3))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = GPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, shape=(m.N, D))

        q = create_Gaussian_meanfield(model=m, observed=[m.Y])

        infr = GradBasedInference(
            inference_algorithm=StochasticVariationalInference(
                model=m, posterior=q, num_samples=10, observed=[m.Y]))
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
            algorithm=GPRegressionSamplingPrediction(
                gp._module_graph, gp._extra_graphs[0], [gp._module_graph.X]),
            alg_name='gp_predict')
        gp.gp_predict.diagonal_variance = False
        gp.gp_predict.jitter = 1e-6

        infr_pred2 = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params)
        xt = np.random.rand(13, 3)
        res = infr_pred2.run(X=mx.nd.array(xt, dtype=dtype))[0]

    def test_prediction_print(self):
        D, X, Y, noise_var, lengthscale, variance = self.gen_data()
        Xt = np.random.rand(20, 3)

        m_gpy = GPy.models.GPRegression(X=X, Y=Y, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), noise_var=noise_var)

        dtype = 'float64'
        m = self.gen_mxfusion_model(dtype, D, noise_var, lengthscale, variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)


        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))
        print = infr.print_params()
        assert (len(print) > 1)

    def test_module_clone(self):
        D, X, Y, noise_var, lengthscale, variance = self.gen_data()
        dtype = 'float64'

        m = Model()
        m.N = Variable()
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = GPRegression.define_variable(X=mx.nd.zeros((2, 3)), kernel=kernel, noise_var=mx.nd.ones((1,)), dtype=dtype)
        m.clone()
