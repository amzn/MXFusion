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
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.components.functions import MXFusionGluonFunction
from mxfusion.common.config import get_default_dtype


class InferenceSerializationTests(unittest.TestCase):
    """
    Test class that tests the MXFusion.utils methods.
    """

    def remove_saved_files(self, prefix):
        import os, glob
        for filename in glob.glob(prefix+"*"):
            os.remove(filename)

    def setUp(self):
        self.PREFIX = 'test_' + str(uuid.uuid4())

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
            v.set_prior(mf.components.distributions.Normal(mean=mx.nd.array([0], dtype=dtype), variance=m.prior_variance))
        m.y = mf.components.distributions.Normal.define_variable(mean=m.r, variance=m.v, shape=(m.N,1))

        return m

    def make_net(self):
        D = 100
        dtype = get_default_dtype()
        net = nn.HybridSequential(prefix='hybrid0_')
        with net.name_scope():
            net.add(nn.Dense(D, activation="tanh", dtype=dtype))
            net.add(nn.Dense(D, activation="tanh", dtype=dtype))
            net.add(nn.Dense(1, flatten=True, dtype=dtype))
        net.initialize(mx.init.Xavier(magnitude=3))
        return net

    def test_meanfield_saving(self):
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

        infr.save(prefix=self.PREFIX)
        self.remove_saved_files(self.PREFIX)

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

        infr.save(prefix=self.PREFIX)

        net2 = self.make_net()
        net2(x_nd)

        m2 = self.make_model(net2)

        observed2 = [m2.y, m2.x]
        q2 = create_Gaussian_meanfield(model=m2, observed=observed2)
        alg2 = StochasticVariationalInference(num_samples=3, model=m2, observed=observed2, posterior=q2)
        infr2 = GradBasedInference(inference_algorithm=alg2, grad_loop=BatchInferenceLoop())
        infr2.initialize(y=y_nd, x=x_nd)

        # Load previous parameters
        infr2.load(graphs_file=self.PREFIX+'_graphs.json',
                   parameters_file=self.PREFIX+'_params.json',
                   inference_configuration_file=self.PREFIX+'_configuration.json',
                   mxnet_constants_file=self.PREFIX+'_mxnet_constants.json',
                   variable_constants_file=self.PREFIX+'_variable_constants.json')

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
        self.remove_saved_files(self.PREFIX)
