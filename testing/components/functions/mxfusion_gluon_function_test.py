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
import numpy as np
import mxnet.gluon.nn as nn
import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.initializer import Zero
from mxfusion.components.functions.mxfusion_gluon_function import MXFusionGluonFunction
from mxfusion.components import Variable
from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples
from mxfusion import Model
from mxfusion.inference import Inference, ForwardSamplingAlgorithm


@pytest.mark.usefixtures("set_seed")
class TestMXFusionGluonFunctionTests(object):
    """
    Tests the MXFusion.core.factor_graph.FactorGraph class.
    """
    def setUp(self):
        self.D = 10
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(nn.Dense(self.D, in_units=1, activation="relu"))
        self.net.initialize()

    def _make_gluon_function_evaluation(self, dtype, broadcastable):
        class Dot(HybridBlock):
            def hybrid_forward(self, F, a, b):
                return F.linalg.gemm2(a, b)

        dot = Dot(prefix='dot')
        dot.initialize()
        dot.hybridize()
        func_wrapper = MXFusionGluonFunction(dot, 1, dtype=dtype,
                                             broadcastable=broadcastable)
        out = func_wrapper(Variable(shape=(3, 4)), Variable(shape=(4, 5)))

        return out.factor

    @pytest.mark.parametrize("dtype, A, A_isSamples, B, B_isSamples, num_samples, broadcastable", [
        (np.float64, np.random.rand(2,3,4), True, np.random.rand(4,5), False, 2, True),
        (np.float64, np.random.rand(2,3,4), True, np.random.rand(2,4,5), True, 2, True),
        (np.float64, np.random.rand(3,4), False, np.random.rand(4,5), False, 0, True),
        (np.float32, np.random.rand(2,3,4), True, np.random.rand(4,5), False, 2, False),
        (np.float32, np.random.rand(2,3,4), True, np.random.rand(2,4,5), True, 2, False),
        (np.float32, np.random.rand(3,4), False, np.random.rand(4,5), False, 0, False)
        ])
    def test_eval(self, dtype, A, A_isSamples, B, B_isSamples, num_samples,
                  broadcastable):

        np_isSamples = A_isSamples or B_isSamples
        if np_isSamples:
            if not A_isSamples:
                A_np = np.expand_dims(A, axis=0)
            else:
                A_np = A
            if not B_isSamples:
                B_np = np.expand_dims(B, axis=0)
            else:
                B_np = B
            res_np = np.einsum('ijk, ikh -> ijh', A_np, B_np)
        else:
            res_np = A.dot(B)

        eval = self._make_gluon_function_evaluation(dtype, broadcastable)
        A_mx = mx.nd.array(A, dtype=dtype)
        if not A_isSamples:
            A_mx = add_sample_dimension(mx.nd, A_mx)
        B_mx = mx.nd.array(B, dtype=dtype)
        if not B_isSamples:
            B_mx = add_sample_dimension(mx.nd, B_mx)
        variables = {eval.dot_input_0.uuid: A_mx, eval.dot_input_1.uuid: B_mx}
        res_rt = eval.eval(F=mx.nd, variables=variables)

        assert np_isSamples == array_has_samples(mx.nd, res_rt)
        assert np.allclose(res_np, res_rt.asnumpy())

    def _make_gluon_function_evaluation_rand_param(self, dtype, broadcastable):
        class Dot(HybridBlock):
            def __init__(self, params=None, prefix=None):
                super(Dot, self).__init__(params=params, prefix=prefix)
                with self.name_scope():
                    self.const = self.params.get('const', shape=(1,), dtype=dtype, init=Zero())

            def hybrid_forward(self, F, a, b, const):
                return F.broadcast_add(F.linalg.gemm2(a, b), const)

        dot = Dot(prefix='dot_')
        dot.initialize()
        dot.hybridize()
        print(dot.collect_params())
        func_wrapper = MXFusionGluonFunction(dot, 1, dtype=dtype,
                                             broadcastable=broadcastable)
        from mxfusion.components.distributions.normal import Normal
        rand_var = Normal.define_variable(shape=(1,))
        out = func_wrapper(Variable(shape=(3, 4)), Variable(shape=(4, 5)), dot_const=rand_var)

        return out.factor

    @pytest.mark.parametrize("dtype, A, A_isSamples, B, B_isSamples, C, C_isSamples, num_samples, broadcastable", [
        (np.float64, np.random.rand(2,3,4), True, np.random.rand(4,5), False, np.random.rand(1), False, 2, True),
        (np.float64, np.random.rand(2,3,4), True, np.random.rand(2,4,5), True, np.random.rand(2, 1), True, 2, True),
        (np.float64, np.random.rand(3,4), False, np.random.rand(4,5), False, np.random.rand(2, 1), True, 2, True)
        ])
    def test_eval_gluon_parameters(self, dtype, A, A_isSamples, B,
                                   B_isSamples, C, C_isSamples, num_samples, broadcastable):

        np_isSamples = A_isSamples or B_isSamples
        if np_isSamples:
            if not A_isSamples:
                A_np = np.expand_dims(A, axis=0)
            else:
                A_np = A
            if not B_isSamples:
                B_np = np.expand_dims(B, axis=0)
            else:
                B_np = B
            res_np = np.einsum('ijk, ikh -> ijh', A_np, B_np)
            if C_isSamples:
                res_np += C[:,:,None]
            else:
                res_np += C
        else:
            res_np = A.dot(B)
            if C_isSamples:
                res_np = res_np[None,:,:] + C[:,:,None]
            else:
                res_np += C
        np_isSamples = np_isSamples or C_isSamples

        eval = self._make_gluon_function_evaluation_rand_param(dtype, broadcastable)
        A_mx = mx.nd.array(A, dtype=dtype)
        if not A_isSamples:
            A_mx = add_sample_dimension(mx.nd, A_mx)
        B_mx = mx.nd.array(B, dtype=dtype)
        if not B_isSamples:
            B_mx = add_sample_dimension(mx.nd, B_mx)
        C_mx = mx.nd.array(C, dtype=dtype)
        if not C_isSamples:
            C_mx = add_sample_dimension(mx.nd, C_mx)
        variables = {eval.dot_input_0.uuid: A_mx, eval.dot_input_1.uuid: B_mx, eval.dot_const.uuid: C_mx}
        res_rt = eval.eval(F=mx.nd, variables=variables)

        assert np_isSamples == array_has_samples(mx.nd, res_rt)
        assert np.allclose(res_np, res_rt.asnumpy())

    def test_success(self):
        self.setUp()
        f = MXFusionGluonFunction(self.net, num_outputs=1)
        x = Variable()
        y = f(x)
        #z = y.value.eval({'x' : mx.nd.ones(self.D)})

    def test_gluon_parameters(self):
        self.setUp()

        m = Model()
        m.x = Variable(shape=(1,1))
        m.f = MXFusionGluonFunction(self.net, num_outputs=1)
        m.y = m.f(m.x)

        infr = Inference(ForwardSamplingAlgorithm(m, observed=[m.x]))
        infr.run(x=mx.nd.ones((1, 1)))
        assert all([v.uuid in infr.params.param_dict for v in m.f.parameters.values()])
