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
from mxfusion.runtime.distributions import WishartRuntime
from mxfusion.components.variables.runtime_variable import add_sample_dimension
np.random.seed(0)
mx.random.seed(0)


@pytest.mark.usefixtures("set_seed")
class TestDirichletRuntimeDistribution(object):

    @pytest.mark.parametrize("dtype, n, V, rv, add_sample_dim", [
        (np.float64, np.random.rand(1)+3, np.random.rand(3,3), np.random.rand(3,3), True),
        (np.float64, np.random.rand(4,1)+3, np.random.rand(4,3,3), np.random.rand(4,3,3), False),
        (np.float64, np.random.rand(4,1)+3, np.random.rand(4,3,3), np.random.rand(1,3,3), False),
        (np.float32, np.random.rand(4,1)+3, np.random.rand(4,3,3), np.random.rand(4,3,3), False)
        ])
    def test_log_pdf(self, dtype, n, V, rv, add_sample_dim):
        V = (np.expand_dims(V, -2)*np.expand_dims(V, -3)).sum(-1)
        rv = (np.expand_dims(rv, -2)*np.expand_dims(rv, -3)).sum(-1)

        from scipy.stats import wishart
        if n.ndim > 1:
            if rv.shape[0]<V.shape[0]:
                rv_r = np.repeat(rv, V.shape[0], 0)
            else:
                rv_r = rv
            log_pdf_np = np.array([wishart.logpdf(rv_r[i], n[i,0], V[i]) for i in range(V.shape[0])])
        else:
            log_pdf_np = wishart.logpdf(rv, n[0], V)

        if add_sample_dim:
            n_mx = add_sample_dimension(mx.nd, mx.nd.array(n, dtype=dtype))
            V_mx = add_sample_dimension(mx.nd, mx.nd.array(V, dtype=dtype))
            rv_mx = add_sample_dimension(mx.nd, mx.nd.array(rv, dtype=dtype))
        else:
            n_mx = mx.nd.array(n, dtype=dtype)
            V_mx = mx.nd.array(V, dtype=dtype)
            rv_mx = mx.nd.array(rv, dtype=dtype)
        wishart_dist = WishartRuntime(degrees_of_freedom=n_mx, scale=V_mx)
        log_pdf_rt = wishart_dist.log_pdf(rv_mx)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-3, 1e-4
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)

    @pytest.mark.skip(reason='TODO Sampling from gamma currently fails on linux builds with the MXNet pre release build required for certain functionality in the runtime distribution changes')
    def test_draw_samples(self):
        np.random.seed(0)
        mx.random.seed(0)
        num_samples = 1000

        n_mx = mx.nd.array(np.random.rand(1, 1), dtype='float64')+2
        V = np.random.rand(1, 2, 2)
        V = (np.expand_dims(V, -2)*np.expand_dims(V, -3)).sum(-1)
        V_mx = mx.nd.array(V, dtype='float64')
        wishart_dist = WishartRuntime(degrees_of_freedom=n_mx, scale=V_mx)
        samples = wishart_dist.draw_samples(num_samples).asnumpy()
        rtol, atol = 1e-1, 1e-1
        assert np.allclose(samples.mean(0), wishart_dist.mean.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(samples.var(0), wishart_dist.variance.asnumpy(), rtol=rtol, atol=atol)

    @pytest.mark.skip(reason='TODO Sampling from gamma currently fails on linux builds with the MXNet pre release build required for certain functionality in the runtime distribution changes')
    def test_draw_samples_broadcast(self):
        np.random.seed(0)
        mx.random.seed(0)
        num_samples = 10

        n_mx = mx.nd.array(np.random.rand(num_samples, 1), dtype='float64')+2
        V = np.random.rand(num_samples, 2, 2)
        V = (np.expand_dims(V, -2)*np.expand_dims(V, -3)).sum(-1)
        V_mx = mx.nd.array(V, dtype='float64')
        wishart_dist = WishartRuntime(degrees_of_freedom=n_mx, scale=V_mx)
        samples = wishart_dist.draw_samples(num_samples)
        assert samples.shape == (num_samples, 2, 2)
