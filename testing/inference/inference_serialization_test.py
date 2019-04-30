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
import uuid
import numpy as np
import mxnet as mx
import mxnet.gluon.nn as nn
import mxfusion as mf
import os
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.components.functions import MXFusionGluonFunction
from mxfusion.common.config import get_default_dtype
from mxfusion.components.functions.operators import broadcast_to
from mxfusion import Variable, Model
from mxfusion.inference import Inference, ForwardSamplingAlgorithm


class InferenceSerializationTests(unittest.TestCase):
    """
    Test class that tests the MXFusion.utils methods.
    """

    def setUp(self):
        self.PREFIX = 'test_' + str(uuid.uuid4())
        self.ZIPNAME = self.PREFIX + '_inference.zip'

    def make_model(self, net):
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
        D = 15
        dtype = get_default_dtype()
        net = nn.HybridSequential(prefix='hybrid0_')
        with net.name_scope():
            net.add(nn.Dense(D, activation="tanh", dtype=dtype, in_units=1))
            net.add(nn.Dense(D, activation="tanh", dtype=dtype, in_units=D))
            net.add(nn.Dense(1, dtype=dtype, in_units=D))
        net.initialize(mx.init.Xavier(magnitude=3))
        return net

    def make_simple_gluon_model(self):
        net = self.make_net()
        m = Model()
        m.x = Variable(shape=(1, 1))
        m.f = MXFusionGluonFunction(net, num_outputs=1)
        m.y = m.f(m.x)
        return m

    def make_gpregr_model(self, lengthscale, variance, noise_var):
        from mxfusion.models import Model
        from mxfusion.components.variables import Variable, PositiveTransformation
        from mxfusion.modules.gp_modules import GPRegression
        from mxfusion.components.distributions.gp.kernels import RBF

        dtype = 'float64'
        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.noise_var = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array(noise_var, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True, variance=mx.nd.array(variance, dtype=dtype), lengthscale=mx.nd.array(lengthscale, dtype=dtype), dtype=dtype)
        m.Y = GPRegression.define_variable(X=m.X, kernel=kernel, noise_var=m.noise_var, shape=(m.N, 1), dtype=dtype)
        return m

    def test_meanfield_saving(self):
        dtype = get_default_dtype()
        x = np.random.rand(10, 1)
        y = np.random.rand(10, 1)
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

        infr.save(self.ZIPNAME)
        os.remove(self.ZIPNAME)

    def test_meanfield_save_and_load(self):
        dtype = get_default_dtype()
        from mxfusion.inference.meanfield import create_Gaussian_meanfield
        from mxfusion.inference import StochasticVariationalInference
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import BatchInferenceLoop

        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1)
        x_nd, y_nd = mx.nd.array(y, dtype=dtype), mx.nd.array(x, dtype=dtype)

        net = self.make_net()
        net(x_nd)

        m = self.make_model(net)

        observed = [m.y, m.x]
        q = create_Gaussian_meanfield(model=m, observed=observed)
        alg = StochasticVariationalInference(num_samples=3, model=m, observed=observed, posterior=q)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
        infr.initialize(y=y_nd, x=x_nd)
        infr.run(max_iter=1, learning_rate=1e-2, y=y_nd, x=x_nd)

        infr.save(self.ZIPNAME)

        net2 = self.make_net()
        net2(x_nd)

        m2 = self.make_model(net2)

        observed2 = [m2.y, m2.x]
        q2 = create_Gaussian_meanfield(model=m2, observed=observed2)
        alg2 = StochasticVariationalInference(num_samples=3, model=m2, observed=observed2, posterior=q2)
        infr2 = GradBasedInference(inference_algorithm=alg2, grad_loop=BatchInferenceLoop())
        infr2.initialize(y=y_nd, x=x_nd)

        # Load previous parameters
        infr2.load(self.ZIPNAME)

        for original_uuid, original_param in infr.params.param_dict.items():
            original_data = original_param.data().asnumpy()
            reloaded_data = infr2.params.param_dict[infr2._uuid_map[original_uuid]].data().asnumpy()
            assert np.all(np.isclose(original_data, reloaded_data))

        for original_uuid, original_param in infr.params.constants.items():
            if isinstance(original_param, mx.ndarray.ndarray.NDArray):
                original_data = original_param.asnumpy()
                reloaded_data = infr2.params.constants[infr2._uuid_map[original_uuid]].asnumpy()
            else:
                original_data = original_param
                reloaded_data = infr2.params.constants[infr2._uuid_map[original_uuid]]

            assert np.all(np.isclose(original_data, reloaded_data))

        infr2.run(max_iter=1, learning_rate=1e-2, y=y_nd, x=x_nd)
        os.remove(self.ZIPNAME)


    def test_gp_module_save_and_load(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        Xt = np.random.rand(20, 3)
        Y = np.random.rand(10, 1)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)
        variance = np.random.rand(1)
        dtype = 'float64'
        m = self.make_gpregr_model(lengthscale, variance, noise_var)

        observed = [m.X, m.Y]
        from mxfusion.inference import MAP, Inference
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        infr.save(self.ZIPNAME)


        m2 = self.make_gpregr_model(lengthscale, variance, noise_var)

        observed2 = [m2.X, m2.Y]
        infr2 = Inference(MAP(model=m2, observed=observed2), dtype=dtype)
        infr2.initialize(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        # Load previous parameters
        infr2.load(self.ZIPNAME)

        for original_uuid, original_param in infr.params.param_dict.items():
            original_data = original_param.data().asnumpy()
            reloaded_data = infr2.params.param_dict[infr2._uuid_map[original_uuid]].data().asnumpy()
            assert np.all(np.isclose(original_data, reloaded_data))

        for original_uuid, original_param in infr.params.constants.items():
            if isinstance(original_param, mx.ndarray.ndarray.NDArray):
                original_data = original_param.asnumpy()
                reloaded_data = infr2.params.constants[infr2._uuid_map[original_uuid]].asnumpy()
            else:
                original_data = original_param
                reloaded_data = infr2.params.constants[infr2._uuid_map[original_uuid]]

            assert np.all(np.isclose(original_data, reloaded_data))

        loss2, _ = infr2.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        os.remove(self.ZIPNAME)

    def test_gluon_func_save_and_load(self):
        m = self.make_simple_gluon_model()
        infr = Inference(ForwardSamplingAlgorithm(m, observed=[m.x]))
        infr.run(x=mx.nd.ones((1, 1)))
        infr.save(self.ZIPNAME)

        m2 = self.make_simple_gluon_model()
        infr2 = Inference(ForwardSamplingAlgorithm(m2, observed=[m2.x]))
        infr2.run(x=mx.nd.ones((1, 1)))
        infr2.load(self.ZIPNAME)
        infr2.run(x=mx.nd.ones((1, 1)))

        for n in m.f.parameter_names:
            assert np.allclose(infr.params[getattr(m.y.factor, n)].asnumpy(), infr2.params[getattr(m2.y.factor, n)].asnumpy())

        os.remove(self.ZIPNAME)
