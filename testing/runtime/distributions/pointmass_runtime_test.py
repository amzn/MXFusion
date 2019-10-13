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
from mxfusion.runtime.distributions import PointMassRuntime
from mxfusion.components.variables.runtime_variable import add_sample_dimension


@pytest.mark.usefixtures("set_seed")
class TestNormalRuntimeDistribution(object):

    @pytest.mark.parametrize("dtype, loc", [
        (np.float64, np.random.rand(2))
        ])
    def test_log_pdf(self, dtype, loc):

        loc_mx = add_sample_dimension(mx.nd, mx.nd.array(loc, dtype=dtype))
        dist = PointMassRuntime(location=loc_mx)
        log_pdf_rt = dist.log_pdf(loc_mx).asnumpy()

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(np.zeros_like(log_pdf_rt), log_pdf_rt, rtol=rtol, atol=atol)

    def test_draw_samples(self):
        np.random.seed(0)
        mx.random.seed(0)
        num_samples = 10

        loc_mx = mx.nd.array(np.random.rand(1, 2), dtype='float64')
        dist = PointMassRuntime(location=loc_mx)
        samples = dist.draw_samples(num_samples).asnumpy()
        rtol, atol = 1e-1, 1e-1
        assert np.allclose(samples.mean(0), dist.mean.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(np.zeros_like(samples), dist.variance.asnumpy(), rtol=rtol, atol=atol)

    def test_draw_samples_broadcast(self):
        num_samples = 10

        loc_mx = mx.nd.array(np.random.rand(num_samples, 2), dtype='float64')
        dist = PointMassRuntime(location=loc_mx)
        samples = dist.draw_samples(num_samples).asnumpy()
        assert samples.shape == (num_samples, 2)
