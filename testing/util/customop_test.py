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
from mxfusion.util.customop import broadcast_to_w_samples

@pytest.mark.usefixtures("set_seed")
class TestBroadcastToWithSamplesOp(object):
    """
    Tests the custom operator BroadcastToWithSamples
    """

    def make_block(self, isSamples, shape):
        class Testop(mx.gluon.HybridBlock):
            def __init__(self, isSamples, shape, **kw):
                self.isSamples = isSamples
                self.shape = shape
                super(Testop, self).__init__(**kw)

            def hybrid_forward(self, F, x, **kw):
                return broadcast_to_w_samples(F, x, self.shape, self.isSamples)
        return Testop(isSamples, shape)

    @pytest.mark.parametrize("data, isSamples, shape, hybridize", [
        (np.array([[2, 3, 4], [3, 4, 5]]), False, (5, 4, 2, 3), True),
        (np.array([[2, 3, 4], [3, 4, 5]]), True, (2, 4, 5, 3), True),
        (np.array([2, 3, 4]), False, (2, 4, 5, 3), True),
        (np.array([[2, 3, 4], [3, 4, 5]]), False, (5, 4, 2, 3), False),
        (np.array([[2, 3, 4], [3, 4, 5]]), True, (2, 4, 5, 3), False),
        (np.array([2, 3, 4]), False, (2, 4, 5, 3), False)
    ])
    def test_forward(self, data, isSamples, shape, hybridize):
        block = self.make_block(isSamples, shape)
        if hybridize:
            block.hybridize()
        res = block(mx.nd.array(data, dtype=np.float64))
        res_np = np.empty(shape)
        if isSamples:
            res_np[:] = data.reshape(*((data.shape[0],)+(1,)*(len(shape)-len(data.shape))+data.shape[1:]))
        else:
            res_np[:] = data
        assert res.shape == shape
        assert np.all(res_np == res.asnumpy())

    @pytest.mark.parametrize("data, isSamples, shape, w, hybridize", [
        (np.array([[2, 3, 4], [3, 4, 5]]), False, (5, 4, 2, 3), np.random.rand(5, 4, 2, 3), True),
        (np.array([[2, 3, 4], [3, 4, 5]]), True, (2, 4, 5, 3), np.random.rand(2, 4, 5, 3), True),
        (np.array([2, 3, 4]), False, (2, 4, 5, 3), np.random.rand(2, 4, 5, 3), True),
        (np.array([[2, 3, 4], [3, 4, 5]]), False, (5, 4, 2, 3), np.random.rand(5, 4, 2, 3), False),
        (np.array([[2, 3, 4], [3, 4, 5]]), True, (2, 4, 5, 3), np.random.rand(2, 4, 5, 3), False),
        (np.array([2, 3, 4]), False, (2, 4, 5, 3), np.random.rand(2, 4, 5, 3), False)
    ])
    def test_backward(self, data, isSamples, shape, w, hybridize):
        block = self.make_block(isSamples, shape)
        data_mx = mx.nd.array(data, dtype=np.float64)
        data_mx.attach_grad()
        w_mx = mx.nd.array(w, dtype=np.float64)
        with mx.autograd.record():
            b = block(data_mx)
            loss = mx.nd.sum(b * w_mx)
        loss.backward()
        data_grad = data_mx.grad.asnumpy()
        if isSamples:
            grad_np = w.reshape(*((data.shape[0], -1) + data.shape[1:])).sum(1)
        else:
            grad_np = w.reshape(*((-1,) + data.shape)).sum(0)
        assert data_grad.shape == data.shape
        assert np.allclose(data_grad, grad_np)
