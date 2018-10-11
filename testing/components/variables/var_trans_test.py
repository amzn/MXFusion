import pytest
import mxnet as mx
import numpy as np
import numpy.testing as npt
from mxfusion.components.variables.var_trans import PositiveTransformation


@pytest.mark.usefixtures("set_seed")
class TestVariableTransformation(object):
    """
    Tests the MXFusion.core.var_trans file for variable transformations
    """
    def test_softplus(self):
        v_orig = mx.nd.array([-10.], dtype=np.float64)
        p = PositiveTransformation()
        v_pos = p.transform(v_orig)
        v_inv_trans = p.inverseTransform(v_pos)
        assert v_orig.asnumpy()[0] < 0
        assert v_pos.asnumpy()[0] > 0
        assert v_inv_trans.asnumpy()[0] < 0
        npt.assert_allclose(v_inv_trans.asnumpy()[0], v_orig.asnumpy()[0], rtol=1e-7, atol=1e-10)

    @pytest.mark.parametrize("x, rtol, atol", [
        (mx.nd.array([10], dtype=np.float64), 1e-7, 1e-10),
        (mx.nd.array([1e-30], dtype=np.float64), 1e-7, 1e-10),
        (mx.nd.array([5], dtype=np.float32), 1e-4, 1e-5),
        (mx.nd.array([1e-6], dtype=np.float32), 1e-4, 1e-5)
    ])
    def test_softplus_numerical(self, x, rtol, atol):

        p = PositiveTransformation()
        mf_pos = p.transform(x)
        mf_inv = p.inverseTransform(mf_pos)

        np_pos = np.log1p(np.exp(x.asnumpy()))
        np_inv = np.log(np.expm1(np_pos))

        npt.assert_allclose(mf_pos.asnumpy(), np_pos, rtol=rtol, atol=atol)
        npt.assert_allclose(mf_inv.asnumpy(), np_inv, rtol=rtol, atol=atol)
        npt.assert_allclose(mf_inv.asnumpy(), x.asnumpy(), rtol=rtol, atol=atol)
