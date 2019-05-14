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


import mxnet as mx
import mxnet.gluon.nn as nn
import numpy as np
import pytest
import mxfusion as mf
from mxfusion import Model, Posterior, Variable
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.components.functions import MXFusionGluonFunction
from mxfusion.util.testutils import make_basic_model
from mxfusion.inference import ScoreFunctionInference, ScoreFunctionRBInference, StochasticVariationalInference
from mxfusion.common.config import get_default_dtype
from mxfusion.components.functions.operators import broadcast_to
from mxfusion import Variable


@pytest.mark.usefixtures("set_seed")
class TestScoreFunction(object):
    """
    Test class that tests the MXFusion.inference.score_function classes.
    """

    def make_bnn_model(self, net):
        dtype = get_default_dtype()
        m = mf.models.Model(verbose=True)
        m.N = mf.components.Variable()
        m.f = MXFusionGluonFunction(net, num_outputs=1)
        m.x = mf.components.Variable(shape=(m.N,1))
        m.v = mf.components.Variable(shape=(1,), transformation=PositiveTransformation(), initial_value=0.01)
        m.prior_variance = mf.components.Variable(shape=(1,), transformation=PositiveTransformation())
        m.r = m.f(m.x)
        for _, v in m.r.factor.parameters.items():
            mean = broadcast_to(Variable(mx.nd.array([0], dtype=dtype)),
                                v.shape)
            var = broadcast_to(m.prior_variance, v.shape)
            v.set_prior(mf.components.distributions.Normal(mean=mean, variance=var))
        m.y = mf.components.distributions.Normal.define_variable(mean=m.r, variance=broadcast_to(m.v, (m.N, 1)), shape=(m.N, 1))
        return m

    def make_net(self):
        dtype = get_default_dtype()
        D = 15
        net = nn.HybridSequential(prefix='hybrid0_')
        with net.name_scope():
            net.add(nn.Dense(D, activation="tanh", dtype=dtype, in_units=1))
            net.add(nn.Dense(D, activation="tanh", dtype=dtype, in_units=D))
            net.add(nn.Dense(1, dtype=dtype, in_units=D))
        net.initialize(mx.init.Xavier(magnitude=3))
        return net

    def make_ppca_data(self):
        def log_spiral(a,b,t):
            x = a * np.exp(b*t) * np.cos(t)
            y = a * np.exp(b*t) * np.sin(t)
            return np.vstack([x,y]).T

        a = 1
        b = 0.1
        t = np.linspace(0,6*np.pi,self.N)
        r = log_spiral(a,b,t)
        w = np.random.randn(self.K,self.N)
        x_train = np.dot(r,w) + np.random.randn(self.N,self.N) * 1e-3
        return x_train

    def make_ppca_model(self):
        dtype = get_default_dtype()
        m = Model()
        m.w = Variable(shape=(self.K,self.D), initial_value=mx.nd.array(np.random.randn(self.K,self.D)))
        dot = nn.HybridLambda(function='dot')
        m.dot = mf.functions.MXFusionGluonFunction(dot, num_outputs=1, broadcastable=False)
        cov = mx.nd.broadcast_to(mx.nd.expand_dims(mx.nd.array(np.eye(self.K,self.K), dtype=dtype), 0),shape=(self.N,self.K,self.K))
        m.z = mf.distributions.MultivariateNormal.define_variable(mean=mx.nd.zeros(shape=(self.N,self.K), dtype=dtype), covariance=cov, shape=(self.N,self.K))
        sigma_2 = Variable(shape=(1,), transformation=PositiveTransformation())
        m.x = mf.distributions.Normal.define_variable(mean=m.dot(m.z, m.w), variance=sigma_2, shape=(self.N,self.D))
        return m

    def make_ppca_post(self, m):
        from mxfusion.inference import BatchInferenceLoop, GradBasedInference
        dtype = get_default_dtype()
        class SymmetricMatrix(mx.gluon.HybridBlock):
            def hybrid_forward(self, F, x, *args, **kwargs):
                return F.sum((F.expand_dims(x, 3)*F.expand_dims(x, 2)), axis=-3)
        q = mf.models.Posterior(m)
        sym = mf.components.functions.MXFusionGluonFunction(SymmetricMatrix(), num_outputs=1, broadcastable=False)
        cov = Variable(shape=(self.N,self.K,self.K), initial_value=mx.nd.broadcast_to(mx.nd.expand_dims(mx.nd.array(np.eye(self.K,self.K) * 1e-2, dtype=dtype), 0),shape=(self.N,self.K,self.K)))
        q.post_cov = sym(cov)
        q.post_mean = Variable(shape=(self.N,self.K), initial_value=mx.nd.array(np.random.randn(self.N,self.K), dtype=dtype))
        q.z.set_prior(mf.distributions.MultivariateNormal(mean=q.post_mean, covariance=q.post_cov))
        return q

    def get_ppca_grad(self, x_train, inf_type, num_samples=100):
        import random
        dtype = get_default_dtype()
        random.seed(0)
        np.random.seed(0)
        mx.random.seed(0)
        m = self.make_ppca_model()
        q = self.make_ppca_post(m)
        observed = [m.x]
        alg = inf_type(num_samples=num_samples, model=m, posterior=q, observed=observed)

        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import BatchInferenceLoop

        infr = GradBasedInference(inference_algorithm=alg,  grad_loop=BatchInferenceLoop())
        infr.initialize(x=mx.nd.array(x_train, dtype=dtype))
        infr.run(max_iter=1, learning_rate=1e-2, x=mx.nd.array(x_train, dtype=dtype), verbose=False)
        return infr, q.post_mean

    def test_score_function_batch(self):
        dtype = get_default_dtype()
        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1)
        x_nd, y_nd = mx.nd.array(y, dtype=dtype), mx.nd.array(x, dtype=dtype)

        self.net = self.make_net()
        self.net(x_nd)

        m = self.make_bnn_model(self.net)

        from mxfusion.inference.meanfield import create_Gaussian_meanfield
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import BatchInferenceLoop
        observed = [m.y, m.x]
        q = create_Gaussian_meanfield(model=m, observed=observed)
        alg = ScoreFunctionInference(num_samples=3, model=m, observed=observed, posterior=q)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
        infr.initialize(y=y_nd, x=x_nd)
        infr.run(max_iter=1, learning_rate=1e-2, y=y_nd, x=x_nd)


    def test_score_function_minibatch(self):
        dtype = get_default_dtype()
        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1)
        x_nd, y_nd = mx.nd.array(y, dtype=dtype), mx.nd.array(x, dtype=dtype)

        self.net = self.make_net()
        self.net(x_nd)

        m = self.make_bnn_model(self.net)

        from mxfusion.inference.meanfield import create_Gaussian_meanfield
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import MinibatchInferenceLoop
        observed = [m.y, m.x]
        q = create_Gaussian_meanfield(model=m, observed=observed)
        alg = ScoreFunctionInference(num_samples=3, model=m, observed=observed, posterior=q)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=MinibatchInferenceLoop(batch_size=100, rv_scaling={m.y: 10}))

        infr.initialize(y=(100, 1), x=(100, 1))
        infr.run(max_iter=1, learning_rate=1e-2, y=y_nd, x=x_nd)


    def test_score_function_rb_batch(self):
        dtype = get_default_dtype()
        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1)
        x_nd, y_nd = mx.nd.array(y, dtype=dtype), mx.nd.array(x, dtype=dtype)

        self.net = self.make_net()
        self.net(x_nd)

        m = self.make_bnn_model(self.net)

        from mxfusion.inference.meanfield import create_Gaussian_meanfield
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import BatchInferenceLoop
        observed = [m.y, m.x]
        q = create_Gaussian_meanfield(model=m, observed=observed)
        alg = ScoreFunctionRBInference(num_samples=3, model=m, observed=observed, posterior=q)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
        infr.initialize(y=y_nd, x=x_nd)
        infr.run(max_iter=1, learning_rate=1e-2, y=y_nd, x=x_nd)

    def test_score_function_rb_minibatch(self):
        dtype = get_default_dtype()
        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1)
        x_nd, y_nd = mx.nd.array(y, dtype=dtype), mx.nd.array(x, dtype=dtype)

        self.net = self.make_net()
        self.net(x_nd)

        m = self.make_bnn_model(self.net)

        from mxfusion.inference.meanfield import create_Gaussian_meanfield
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import MinibatchInferenceLoop
        observed = [m.y, m.x]
        q = create_Gaussian_meanfield(model=m, observed=observed)
        alg = ScoreFunctionRBInference(num_samples=3, model=m, observed=observed, posterior=q)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=MinibatchInferenceLoop(batch_size=100, rv_scaling={m.y: 10}))

        infr.initialize(y=(100, 1), x=(100, 1))
        infr.run(max_iter=1, learning_rate=1e-2, y=y_nd, x=x_nd)

    @pytest.mark.parametrize("test_infr, truth_infr, num_samples, N, D, K", [
        (ScoreFunctionInference, StochasticVariationalInference, 100, 100, 100, 2),
        (ScoreFunctionRBInference, StochasticVariationalInference, 100, 100, 100, 2),
        (ScoreFunctionRBInference, ScoreFunctionInference, 100, 100, 100, 2),
        ])
    def test_score_function_gradient(self, test_infr, truth_infr, num_samples, N, D, K):

        self.N = N
        self.D = D
        self.K = K

        x = self.make_ppca_data()
        test_infr, test_mean = self.get_ppca_grad(x, test_infr, num_samples)
        truth_infr, truth_mean = self.get_ppca_grad(x, truth_infr, num_samples)

        test_np = test_infr.params[test_mean].grad.asnumpy()
        truth_np = truth_infr.params[truth_mean].grad.asnumpy()
        # import pdb; pdb.set_trace()
        assert np.allclose(test_np, truth_np, atol=1., rtol=1e-3)
