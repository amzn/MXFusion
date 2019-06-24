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
from mxfusion.runtime.distributions import LaplaceRuntime
from mxfusion.components.variables.runtime_variable import add_sample_dimension


@pytest.mark.usefixtures("set_seed")
class TestLaplaceRuntimeDistribution(object):

    @pytest.mark.parametrize("dtype, loc, scale, rv", [
        (np.float64, np.random.rand(2), np.random.rand(2)+0.1, np.random.rand(2)),
        (np.float64, np.random.rand(3,2), np.random.rand(3,2)+0.1, np.random.rand(3,2)),
        (np.float32, np.random.rand(3,2), np.random.rand(3,2)+0.1, np.random.rand(3,2))
        ])
    def test_log_pdf(self, dtype, loc, scale, rv):
        from scipy.stats import laplace
        log_pdf_np = laplace.logpdf(rv, loc, scale)

        loc_mx = add_sample_dimension(mx.nd, mx.nd.array(loc, dtype=dtype))
        scale_mx = add_sample_dimension(mx.nd, mx.nd.array(scale, dtype=dtype))
        rv_mx = add_sample_dimension(mx.nd, mx.nd.array(rv, dtype=dtype))
        laplace_dist = LaplaceRuntime(location=loc_mx, scale=scale_mx)
        log_pdf_rt = laplace_dist.log_pdf(rv_mx)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)

    def test_draw_samples(self):
        np.random.seed(0)
        mx.random.seed(0)
        num_samples = 1000

        loc_mx = mx.nd.array(np.random.rand(1, 2), dtype='float64')
        scale_mx = mx.nd.array(np.random.rand(1, 2)+0.01, dtype='float64')
        laplace_dist = LaplaceRuntime(location=loc_mx, scale=scale_mx)
        samples = laplace_dist.draw_samples(num_samples)
        rtol, atol = 1e-1, 1e-1
        assert np.allclose(samples.asnumpy().mean(0), laplace_dist.mean.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(samples.asnumpy().var(0), laplace_dist.variance.asnumpy(), rtol=rtol, atol=atol)
