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
import numpy.testing as npt
from mxfusion.components.variables.var_trans import PositiveTransformation, Logistic


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

    @pytest.mark.parametrize("x, upper, lower, rtol, atol", [
        (mx.nd.array([10], dtype=np.float64), 2, 20, 1e-7, 1e-10),
        (mx.nd.array([1e-3], dtype=np.float64), 1e-6, 1e-2, 1e-7, 1e-10),
        (mx.nd.array([1], dtype=np.float32), 1, 200000, 1e-4, 1e-5),
        (mx.nd.array([5], dtype=np.float32), 2, 10000, 1e-4, 1e-5)
    ])
    def test_logistic(self, x, upper, lower, rtol, atol):
        transform = Logistic(upper, lower)
        x_trans = transform.transform(x)
        x_inversed = transform.inverseTransform(x_trans)
        assert x_inversed.dtype == x.dtype
        assert np.isclose(x.asnumpy(), x_inversed.asnumpy(), rtol=rtol, atol=atol)
