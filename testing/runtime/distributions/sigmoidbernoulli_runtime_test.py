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
from mxfusion.runtime.distributions import SigmoidBernoulliRuntime
from mxfusion.components.variables.runtime_variable import add_sample_dimension


@pytest.mark.usefixtures("set_seed")
class TestSigmoidBernoulliRuntimeDistribution(object):

    @pytest.mark.parametrize("dtype, prob, rv", [
        (np.float64, np.random.rand(2), np.random.rand(2)<0.5),
        (np.float64, np.random.rand(3,2), np.random.rand(3,2)<0.5),
        (np.float32, np.random.rand(3,2), np.random.rand(3,2)<0.5)
        ])
    def test_log_pdf(self, dtype, prob, rv):
        from scipy.stats import bernoulli
        log_pdf_np = bernoulli.logpmf(rv, p=1./(1.+np.exp(-prob)))

        prob_mx = add_sample_dimension(mx.nd, mx.nd.array(prob, dtype=dtype))
        rv_mx = add_sample_dimension(mx.nd, mx.nd.array(rv, dtype=dtype))
        sb_dist = SigmoidBernoulliRuntime(prob_true=prob_mx)
        log_pdf_rt = sb_dist.log_pdf(rv_mx)

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

        prob_mx = mx.nd.array(np.random.rand(1, 2), dtype='float64')
        sb_dist = SigmoidBernoulliRuntime(prob_true=prob_mx)
        samples = sb_dist.draw_samples(num_samples)
        rtol, atol = 1e-1, 1e-1
        assert np.allclose(samples.asnumpy().mean(0), sb_dist.mean.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(samples.asnumpy().var(0), sb_dist.variance.asnumpy(), rtol=rtol, atol=atol)
