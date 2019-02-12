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


import unittest
import numpy as np
import mxnet as mx
import mxnet.gluon.nn as nn
import mxfusion as mf
from mxfusion.inference.mh_sampling import MCMCInference, MetropolisHastingsAlgorithm
from mxfusion.components.functions import MXFusionGluonFunction
from mxfusion.common.config import get_default_dtype
from mxfusion.models import Model
from mxfusion.components.distributions import Normal


class MetropolisHastingsTests(unittest.TestCase):
    """
    Test class that tests the MetropolisHastings and MCMC methods.
    """

    def make_simple_model(self, N=100):
        dtype = get_default_dtype()
        m = Model()
        m.mu = Normal.define_variable(mean=mx.nd.array([0], dtype=dtype),
                                      variance=mx.nd.array([100], dtype=dtype), shape=(1,))

        m.s_hat = Normal.define_variable(mean=mx.nd.array([5], dtype=dtype),
                                         variance=mx.nd.array([100], dtype=dtype),
                                         shape=(1,), dtype=dtype)
        trans_mxnet = mx.gluon.nn.HybridLambda(lambda F, x: F.Activation(x, act_type='softrelu'))
        m.trans = MXFusionGluonFunction(trans_mxnet, num_outputs=1, broadcastable=True)
        m.s = m.trans(m.s_hat)
        m.Y = Normal.define_variable(mean=m.mu, variance=m.s, shape=(N,), dtype=dtype)

        return m

    def generate_simple_data(self, N=100):
        mean_groundtruth = 3.
        variance_groundtruth = 5.
        data = np.random.randn(N)*np.sqrt(variance_groundtruth) + mean_groundtruth
        return data


    def test_basic_mhmc_sampling(self):
        dtype = get_default_dtype()
        m = self.make_simple_model()
        data = self.generate_simple_data()

        observed = [m.Y]
        alg = MetropolisHastingsAlgorithm(m, observed, dtype=dtype)
        infr = MCMCInference(alg)
        iterations = 30
        samples = infr.run(Y=mx.nd.array(data, dtype=dtype), max_iter=iterations, verbose=True)

        samples_mu = samples[infr.inference_algorithm.proposal.mu.uuid].asnumpy()[-1]
        parameters_mu = infr.params[infr.inference_algorithm.proposal.mu].asnumpy()
        parameters_mu_prev = infr.params[infr.inference_algorithm.proposal.mu.factor.mean].asnumpy()
        # print("samples_mu : {} \n Proposal : {} \n : z_copy {}\n".format(samples_mu, parameters_mu, parameters_mu_prev))

        self.assertTrue(parameters_mu == samples_mu)
        self.assertTrue(parameters_mu == parameters_mu_prev)

    def make_ppca_model(self, w, noise_var, N, K, D):
        dtype = get_default_dtype()
        from mxfusion.models import Model
        import mxnet.gluon.nn as nn
        from mxfusion.components import Variable
        from mxfusion.components.variables import PositiveTransformation
        m = Model()
        m.w = Variable(shape=(K,D), value=mx.nd.array(w))
        dot = nn.HybridLambda(function='dot')
        m.dot = mf.functions.MXFusionGluonFunction(dot, num_outputs=1, broadcastable=False)
        cov = mx.nd.broadcast_to(mx.nd.expand_dims(mx.nd.array(np.eye(K,K)), 0),shape=(N,K,K))
        m.z = mf.distributions.MultivariateNormal.define_variable(mean=mx.nd.zeros(shape=(N,K)), covariance=cov, shape=(N,K))
        sigma_2 = Variable(shape=(1,), value=mx.nd.array(noise_var))
        m.x = mf.distributions.Normal.define_variable(mean=m.dot(m.z, m.w), variance=sigma_2, shape=(N,D))

        return m

    def generate_ppca_data(self, N, K, D):
        def log_spiral(a,b,t):
            x = a * np.exp(b*t) * np.cos(t)
            y = a * np.exp(b*t) * np.sin(t)
            return np.vstack([x,y]).T

        a = 1
        b = 0.1
        t = np.linspace(0,6*np.pi,N)
        r = log_spiral(a,b,t)
        w = np.random.randn(K,N)
        x_train = np.dot(r,w) + np.random.randn(N,N) * 1e-3
        return x_train, w

    def test_ppca_model_mhmc_sampling(self):
        dtype = get_default_dtype()
        N = 5
        D = 5
        K = 2
        noise_var = mx.nd.array([1.])
        x_train, w = self.generate_ppca_data(N, K, D)
        m = self.make_ppca_model(w, noise_var, N, K, D)

        observed = [m.x]
        alg = MetropolisHastingsAlgorithm(m, observed, variance=mx.nd.array([1e-4]))
        infr = MCMCInference(alg)
        infr.initialize(x=mx.nd.array(x_train))
        iterations = 1000
        samples = infr.run(max_iter=iterations,  x=mx.nd.array(x_train), verbose=True)

        samples_mu = samples[infr.inference_algorithm.proposal.z.uuid].asnumpy()
