import pytest
import mxnet as mx
import numpy as np
from mxfusion import Variable, Model
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
