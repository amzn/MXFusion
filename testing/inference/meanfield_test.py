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
import mxfusion as mf
import mxnet as mx
import numpy as np
import mxnet.gluon.nn as nn
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.components.functions import MXFusionGluonFunction
from mxfusion.util.testutils import make_basic_model
from mxfusion.common.config import get_default_dtype
from mxfusion.components.functions.operators import broadcast_to
from mxfusion import Variable


class MeanFieldInferenceTests(unittest.TestCase):
    """
    Test class that tests the MXFusion.utils methods.
    """

    def make_model(self, net):
        dtype = get_default_dtype()
        m = mf.models.Model(verbose=True)
        m.N = mf.components.Variable()
        m.f = MXFusionGluonFunction(net, num_outputs=1)
        m.x = mf.components.Variable(shape=(m.N, 1))
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

    def test_meanfield_batch(self):
        dtype = get_default_dtype()
        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1)
        x_nd, y_nd = mx.nd.array(y, dtype=dtype), mx.nd.array(x, dtype=dtype)

        self.net = self.make_net()
        self.net(x_nd)

        m = self.make_model(self.net)

        from mxfusion.inference.meanfield import create_Gaussian_meanfield
        from mxfusion.inference import StochasticVariationalInference
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import BatchInferenceLoop
        observed = [m.y, m.x]
        q = create_Gaussian_meanfield(model=m, observed=observed)
        alg = StochasticVariationalInference(num_samples=3, model=m, observed=observed, posterior=q)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
        infr.initialize(y=y_nd, x=x_nd)
        infr.run(max_iter=1, learning_rate=1e-2, y=y_nd, x=x_nd)

    def test_meanfield_minibatch(self):
        dtype = get_default_dtype()
        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1)
        x_nd, y_nd = mx.nd.array(y, dtype=dtype), mx.nd.array(x, dtype=dtype)

        self.net = self.make_net()
        self.net(x_nd)

        m = self.make_model(self.net)

        from mxfusion.inference.meanfield import create_Gaussian_meanfield
        from mxfusion.inference import StochasticVariationalInference
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import MinibatchInferenceLoop
        observed = [m.y, m.x]
        q = create_Gaussian_meanfield(model=m, observed=observed)
        alg = StochasticVariationalInference(num_samples=3, model=m, observed=observed, posterior=q)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=MinibatchInferenceLoop(batch_size=100, rv_scaling={m.y: 10}))

        infr.initialize(y=(100, 1), x=(100, 1))
        infr.run(max_iter=1, learning_rate=1e-2, y=y_nd, x=x_nd)

    def test_softplus_in_params(self):

        m = make_basic_model()

        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1)
        x_nd, y_nd = mx.nd.array(y), mx.nd.array(x)

        from mxfusion.inference.meanfield import create_Gaussian_meanfield
        from mxfusion.inference import StochasticVariationalInference
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import BatchInferenceLoop
        observed = [m.x]
        q = create_Gaussian_meanfield(model=m, observed=observed)
        alg = StochasticVariationalInference(num_samples=3, model=m, observed=observed, posterior=q)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())

        infr.initialize(x=x_nd)
        infr.run(max_iter=1, learning_rate=1e-2, x=x_nd)

        uuid_of_pos_var = m.v.uuid
        infr.params._params[uuid_of_pos_var]._data = mx.nd.array([-10])
        raw_value = infr.params._params[uuid_of_pos_var].data()
        transformed_value = infr.params[m.v]
        assert raw_value.asnumpy()[0] < 0 and transformed_value.asnumpy()[0] > 0
