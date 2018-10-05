import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import add_sample_dimension, is_sampled_array


@pytest.mark.usefixtures("set_seed")
class TestOperators(object):

    def _test_operator(self, operator, inputs, properties=None):
        """
        inputs are mx.nd.array
        properties are just the operator properties needed at model def time.
        """
        properties = properties if properties is not None else {}
        from mxfusion import Variable, Model
        m = Model()
        variables = [Variable() for _ in inputs]
        m.r = operator(*variables, **properties)
        vs = [v for v in m.r.factor.inputs]
        variables = {v[1].uuid: inputs[i] for i,v in enumerate(vs)}
        evaluation = m.r.factor.eval(mx.nd, variables=variables)
        return evaluation

    from mxfusion.components.functions.operators import reshape, dot
    @pytest.mark.parametrize("mxf_operator, mxnet_operator, inputs, properties", [
        (reshape, mx.nd.reshape, [mx.nd.array(np.random.rand(1,4))], {'shape':(2,2), 'reverse': False}),
        (dot, mx.nd.dot, [mx.nd.array(np.random.rand(1,4)), mx.nd.array(np.random.rand(1,4))], {}),
        ])
    def test_operators(self, mxf_operator, mxnet_operator, inputs, properties):
        mxf_result = self._test_operator(mxf_operator, inputs, properties)
        inputs_unsampled = [v[0] for v in inputs]
        mxnet_result = mxnet_operator(*inputs_unsampled, **properties)
        assert np.allclose(mxf_result.asnumpy(), mxnet_result.asnumpy()), (mxf_result, mxnet_result)
