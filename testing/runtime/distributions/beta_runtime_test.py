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
from mxfusion.runtime.distributions import BetaRuntime
from mxfusion.components.variables.runtime_variable import add_sample_dimension


@pytest.mark.usefixtures("set_seed")
class TestBetaRuntimeDistribution(object):

    @pytest.mark.parametrize("dtype, a, b, rv", [
        (np.float64, np.random.rand(2), np.random.rand(2)+0.1, np.random.rand(2)),
        (np.float64, np.random.rand(3,2), np.random.rand(3,2)+0.1, np.random.rand(3,2)),
        (np.float32, np.random.rand(3,2), np.random.rand(3,2)+0.1, np.random.rand(3,2))
        ])
    def test_log_pdf(self, dtype, a, b, rv):
        from scipy.stats import beta
        log_pdf_np = beta.logpdf(rv, a, b)

        alpha_mx = add_sample_dimension(mx.nd, mx.nd.array(a, dtype=dtype))
        beta_mx = add_sample_dimension(mx.nd, mx.nd.array(b, dtype=dtype))
        rv_mx = add_sample_dimension(mx.nd, mx.nd.array(rv, dtype=dtype))
        beta_dist = BetaRuntime(alpha=alpha_mx, beta=beta_mx)
        log_pdf_rt = beta_dist.log_pdf(rv_mx)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)

    @pytest.mark.skip(reason='Sampling from gamma currently fails on linux builds')
    def test_draw_samples(self):
        np.random.seed(0)
        mx.random.seed(0)
        num_samples = 1000

        alpha = mx.nd.array(np.random.rand(1, 2), dtype='float64')
        beta = mx.nd.array(np.random.rand(1, 2)+0.01, dtype='float64')
        beta_dist = BetaRuntime(alpha=alpha, beta=beta)
        samples = beta_dist.draw_samples(num_samples)
        rtol, atol = 1e-1, 1e-1
        assert np.allclose(samples.asnumpy().mean(0), beta_dist.mean.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(samples.asnumpy().var(0), beta_dist.variance.asnumpy(), rtol=rtol, atol=atol)

    @pytest.mark.skip(reason='Sampling from gamma currently fails on linux builds')
    def test_draw_samples_broadcast(self):
        num_samples = 10

        alpha = mx.nd.array(np.random.rand(num_samples, 2), dtype='float64')
        beta = mx.nd.array(np.random.rand(num_samples, 2)+0.01, dtype='float64')
        beta_dist = BetaRuntime(alpha=alpha, beta=beta)
        samples = beta_dist.draw_samples(num_samples)
        assert samples.shape == (num_samples, 2)
