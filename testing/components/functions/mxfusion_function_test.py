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
import mxnet.gluon.nn as nn
import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import Lambda
from mxnet.initializer import Zero
from mxfusion.components.functions.mxfusion_function import MXFusionFunction
from mxfusion.components import Variable
from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples


class TestMXFusionFunctionTests(unittest.TestCase):
    """
    Tests the MXFusion.core.factor_graph.FactorGraph class.
    """
    def instantialize_base_function(self):
        return MXFusionFunction('test_func')

    def instantialize_customize_function(self):
        class TestFunction(MXFusionFunction):
            def __init__(self, func_name, dtype=None, broadcastable=False):
                super(TestFunction, self).__init__(
                    func_name, dtype, broadcastable)
                self._params = {'C': Variable()}

            def eval(self, F, A, B, C):
                return A+B+C, A*B+C

            @property
            def parameters(self):
                return self._params

            @property
            def input_names(self):
                return ['A', 'B', 'C']

            @property
            def output_names(self):
                return ['out1', 'out2']

            @property
            def parameter_names(self):
                return ['C']
        func_mx = Lambda(lambda A, B, C: (A+B+C, A*B+C))
        return TestFunction('test_func'), func_mx

    def test_raise_errors(self):
        f = self.instantialize_base_function()
        self.assertRaises(NotImplementedError, lambda: f.parameters)
        self.assertRaises(NotImplementedError, lambda: f.input_names)
        self.assertRaises(NotImplementedError, lambda: f.output_names)
        self.assertRaises(NotImplementedError, lambda: f.parameter_names)
        self.assertRaises(NotImplementedError, lambda: f.eval(mx.nd, X=1))

    def test_function_execution(self):
        f, f_mx = self.instantialize_customize_function()
        np.random.seed(0)
        A = mx.nd.array(np.random.rand(1, 3, 2))
        B = mx.nd.array(np.random.rand(1, 3, 2))
        C = mx.nd.array(np.random.rand(1))

        A_mf = Variable(shape=A.shape)
        B_mf = Variable(shape=B.shape)
        outs = f(A_mf, B_mf)
        assert len(outs) == 2
        eval = f(A_mf, B_mf)[0].factor
        variables = {A_mf.uuid: A, B_mf.uuid: B, eval.C.uuid: C}
        res_eval = eval.eval(F=mx.nd, variables=variables)
        res_mx = f_mx(A, B, C)
        assert np.allclose(res_eval[0].asnumpy(), res_mx[0].asnumpy())
        assert np.allclose(res_eval[1].asnumpy(), res_mx[1].asnumpy())
