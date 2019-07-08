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
from mxfusion.runtime.distributions import DirichletRuntime
from mxfusion.components.variables.runtime_variable import add_sample_dimension


@pytest.mark.usefixtures("set_seed")
class TestDirichletRuntimeDistribution(object):

    @pytest.mark.parametrize("dtype, a, rv", [
        (np.float64, np.random.rand(3), np.random.rand(3)),
        (np.float64, np.random.rand(4,3), np.random.rand(4,3)),
        (np.float32, np.random.rand(4,3), np.random.rand(4,3))
        ])
    def test_log_pdf(self, dtype, a, rv):
        rv = rv/np.sum(rv, axis=-1, keepdims=True)
        from scipy.stats import dirichlet
        if a.ndim > 1:
            log_pdf_np = np.array([dirichlet.logpdf(rv[i], a[i]) for i in range(rv.shape[0])])
        else:
            log_pdf_np = dirichlet.logpdf(rv, a)

        alpha_mx = add_sample_dimension(mx.nd, mx.nd.array(a, dtype=dtype))
        rv_mx = add_sample_dimension(mx.nd, mx.nd.array(rv, dtype=dtype))
        dir_dist = DirichletRuntime(alpha=alpha_mx)
        log_pdf_rt = dir_dist.log_pdf(rv_mx)

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
        dir_dist = DirichletRuntime(alpha=alpha)
        samples = dir_dist.draw_samples(num_samples).asnumpy()
        mean = samples.mean(0)
        cov = samples.T.dot(samples)/num_samples - mean[:, None]*mean[None, :]
        rtol, atol = 1e-1, 1e-1
        assert np.allclose(mean, dir_dist.mean.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(samples.var(0), dir_dist.variance.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(cov, dir_dist.covariance.asnumpy(), rtol=rtol, atol=atol)

    @pytest.mark.skip(reason='Sampling from gamma currently fails on linux builds')
    def test_draw_samples_broadcast(self):
        num_samples = 10

        alpha = mx.nd.array(np.random.rand(num_samples, 2), dtype='float64')
        dir_dist = DirichletRuntime(alpha=alpha)
        samples = dir_dist.draw_samples(num_samples)
        assert samples.shape == (num_samples, 2)
