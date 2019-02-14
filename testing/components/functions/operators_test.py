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
from mxfusion import Variable, Model
from mxfusion.common.exceptions import ModelSpecificationError
from mxfusion.components.functions.operators import *


@pytest.mark.usefixtures("set_seed")
class TestOperators(object):

    def _test_operator(self, operator, inputs, properties=None):
        """
        inputs are mx.nd.array
        properties are just the operator properties needed at model def time.
        """
        properties = properties if properties is not None else {}
        m = Model()
        variables = [Variable() for _ in inputs]
        m.r = operator(*variables, **properties)
        vs = [v for v in m.r.factor.inputs]
        variables = {v[1].uuid: inputs[i] for i,v in enumerate(vs)}
        evaluation = m.r.factor.eval(mx.nd, variables=variables)
        return evaluation


    @pytest.mark.parametrize("mxf_operator, mxnet_operator, inputs, properties", [
    (reshape, mx.nd.reshape, [mx.nd.array(np.random.rand(1,4))], {'shape':(2,2), 'reverse': False}),
    ])
    def test_operator_replicate(self, mxf_operator, mxnet_operator, inputs, properties):

        properties = properties if properties is not None else {}
        m = Model()
        variables = [Variable() for _ in inputs]
        m.r = mxf_operator(*variables, **properties)
        vs = [v for v in m.r.factor.inputs]
        variables = {v[1].uuid: inputs[i] for i,v in enumerate(vs)}
        evaluation = m.r.factor.eval(mx.nd, variables=variables)

        r_clone = m.extract_distribution_of(m.r)
        vs = [v for v in r_clone.factor.inputs]
        variables = {v[1].uuid: inputs[i] for i,v in enumerate(vs)}
        evaluation2 = r_clone.factor.eval(mx.nd, variables=variables)

        assert np.allclose(evaluation.asnumpy(), evaluation2.asnumpy()), (evaluation, evaluation2)

    @pytest.mark.parametrize("mxf_operator, mxnet_operator, inputs, case", [
        (add, mx.nd.add, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], "add"),
        (subtract, mx.nd.subtract, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], "sub"),
        (multiply, mx.nd.multiply, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], "mul"),
        (divide, mx.nd.divide, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], "div"),
        (power, mx.nd.power, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], "pow"),
        (transpose, mx.nd.transpose, [mx.nd.array(np.random.rand(1,4))], "transpose"),
        ])
    def test_operators_variable_builtins(self, mxf_operator, mxnet_operator, inputs, case):

        m = Model()
        v1 = Variable()
        v2 = Variable()
        variables = [v1, v2] if len(inputs) > 1 else [v1]
        m.r = mxf_operator(*variables)
        vs = [v for v in m.r.factor.inputs]
        variables_rt = {v[1].uuid: inputs[i] for i,v in enumerate(vs)}
        r_eval = m.r.factor.eval(mx.nd, variables=variables_rt)

        m2 = Model()
        v12 = Variable()
        v22 = Variable()
        variables2 = [v12, v22] if len(inputs) > 1 else [v12]
        if case == "add":
            m2.r = v12 + v22
        elif case == "sub":
            m2.r = v12 - v22
        elif case == "mul":
            m2.r = v12 * v22
        elif case == "div":
            m2.r = v12 / v22
        elif case == "pow":
            m2.r = v12 ** v22
        elif case == "transpose":
            m2.r = transpose(v12)
        vs2 = [v for v in m2.r.factor.inputs]
        variables_rt2 = {v[1].uuid: inputs[i] for i,v in enumerate(vs2)}
        p_eval = m2.r.factor.eval(mx.nd, variables=variables_rt2)

        assert np.allclose(r_eval.asnumpy(), p_eval.asnumpy()), (r_eval, p_eval)


    @pytest.mark.parametrize("mxf_operator, mxnet_operator, inputs, properties", [
        (add, mx.nd.add, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], {}),
        (subtract, mx.nd.subtract, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], {}),
        (multiply, mx.nd.multiply, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], {}),
        (divide, mx.nd.divide, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], {}),
        (power, mx.nd.power, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], {}),

        (square, mx.nd.square, [mx.nd.array(np.random.rand(1,4))],  {}),
        (exp, mx.nd.exp, [mx.nd.array(np.random.rand(1,4))],  {}),
        (log, mx.nd.log, [mx.nd.array(np.random.rand(1,4))],  {}),

        (sum, mx.nd.sum, [mx.nd.array(np.random.rand(1,4))],  {}),
        (mean, mx.nd.mean, [mx.nd.array(np.random.rand(1,4))],  {}),
        (prod, mx.nd.prod, [mx.nd.array(np.random.rand(1,4))],  {}),

        (dot, mx.nd.dot, [mx.nd.array(np.random.rand(1,4,1)), mx.nd.array(np.random.rand(1,1,4))], {}),
        (dot, mx.nd.dot, [mx.nd.array(np.random.rand(1,1,4)), mx.nd.array(np.random.rand(1,4,1))], {}),
        (diag, mx.nd.diag, [mx.nd.array(np.random.rand(1,4,4))],  {}),

        (reshape, mx.nd.reshape, [mx.nd.array(np.random.rand(1,4))], {'shape':(2,2), 'reverse': False}),
        (reshape, mx.nd.reshape, [mx.nd.array(np.random.rand(1,4,4))], {'shape':(1,16), 'reverse': False}),
        (transpose, mx.nd.transpose, [mx.nd.array(np.random.rand(1,4))], {}),
        ])
    def test_operators(self, mxf_operator, mxnet_operator, inputs, properties):
        mxf_result = self._test_operator(mxf_operator, inputs, properties)
        inputs_unsampled = [v[0] for v in inputs]
        mxnet_result = mxnet_operator(*inputs_unsampled, **properties)
        assert np.allclose(mxf_result.asnumpy(), mxnet_result.asnumpy()), (mxf_result, mxnet_result)


    @pytest.mark.parametrize("mxf_operator", [
        (add),
        (reshape),
        ])
    def test_empty_operator(self, mxf_operator):
        with pytest.raises(ModelSpecificationError, message="Operator should fail if not passed the correct arguments.") as excinfo:
            mxf_result = mxf_operator()
        assert excinfo.value is not None
