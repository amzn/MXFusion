import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.functions.operators.mxnet_custom_operators import GammaLn

@pytest.mark.usefixtures("set_seed")
class TestCustomOperators(object):

    @pytest.mark.parametrize("dtype, A", [
        (np.float64, np.random.rand(2,3)),
        (np.float32, np.random.rand(2,3)),
        ])
    def test_gammaln_forward(self, dtype, A):
        import scipy as sp
        op = GammaLn()
        in_data = mx.nd.array(A, dtype=dtype).reshape((1,)+A.shape)
        out_data = mx.nd.empty(shape=(1,) + in_data.shape, dtype=dtype)
        out_data_np = sp.special.gammaln(A)
        op.forward(None, ['write'], in_data, out_data, None)
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5

        assert np.allclose(out_data.asnumpy(), out_data_np, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype, A", [
        (np.float64, np.random.rand(2,3)),
        (np.float32, np.random.rand(2,3)),
        ])
    def test_gammaln_backward(self, dtype, A):
        import scipy as sp
        op = GammaLn()
        in_data = mx.nd.array(A, dtype=dtype).reshape((1,)+A.shape)
        out_data = mx.nd.array(sp.special.gammaln(A), dtype=dtype).reshape((1,)+A.shape)
        out_grad = mx.nd.ones(shape=(1,) + in_data.shape, dtype=dtype)
        in_grad = mx.nd.empty(shape=(1,) + in_data.shape, dtype=dtype)

        in_grad_np = sp.special.digamma(A)
        op.backward(['write'], out_grad, in_data, out_data, in_grad, None)
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5

        # import pdb; pdb.set_trace()
        assert np.allclose(in_grad.asnumpy(), in_grad_np, rtol=rtol, atol=atol)
