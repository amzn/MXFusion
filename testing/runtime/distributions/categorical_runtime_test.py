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
from mxfusion.runtime.distributions import CategoricalRuntime
from mxfusion.components.variables.runtime_variable import add_sample_dimension


@pytest.mark.usefixtures("set_seed")
class TestBernoulliRuntimeDistribution(object):

    @pytest.mark.parametrize("dtype, log_prob, rv, normalization, one_hot_encoding", [
        (np.float64,  np.random.rand(1, 3)+1e-2, np.random.randint(0, 3, size=(1,)), True, False),
        (np.float64, np.random.rand(2, 3)+1e-2, np.random.randint(0, 3, size=(2,)), True, False),
        (np.float64, np.random.rand(2, 3)+1e-2, np.random.randint(0, 3, size=(2,)), True, True),
        (np.float64, np.random.rand(2, 3)+1e-2, np.random.randint(0, 3, size=(2,)), False, False),
        (np.float64, np.random.rand(2, 3)+1e-2, np.random.randint(0, 3, size=(2,)), False, True),
        (np.float32, np.random.rand(2, 3)+1e-2, np.random.randint(0, 3, size=(2,)), True, False)
        ])
    def test_log_pdf(self, dtype, log_prob, rv, normalization, one_hot_encoding):
        num_classes = 3
        one_hot = np.zeros((rv.shape[0], num_classes))
        one_hot[np.arange(rv.shape[0]), rv] = 1

        prob = np.exp(log_prob)
        prob = prob/prob.sum(-1)[:,None]

        from scipy.stats import multinomial
        log_pdf_np = multinomial.logpmf(one_hot, 1, p=prob)

        if normalization:
            log_prob_mx = add_sample_dimension(mx.nd, mx.nd.array(log_prob, dtype=dtype))
        else:
            log_prob_mx = add_sample_dimension(mx.nd, mx.nd.array(np.log(prob), dtype=dtype))
        if one_hot_encoding:
            rv_mx = add_sample_dimension(mx.nd, mx.nd.array(one_hot, dtype=dtype))
        else:
            rv_mx = add_sample_dimension(mx.nd, mx.nd.array(rv, dtype=dtype))
        cate_dist = CategoricalRuntime(log_prob=log_prob_mx, num_classes=num_classes, normalization=normalization, one_hot_encoding=one_hot_encoding)
        log_pdf_rt = cate_dist.log_pdf(rv_mx)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("normalization, one_hot_encoding", [
        (True, False),
        (True, True),
        (False, False),
        (False, True)
        ])
    def test_draw_samples(self, normalization, one_hot_encoding):
        np.random.seed(0)
        mx.random.seed(0)
        num_classes = 3
        num_samples = 1000

        log_prob = np.random.rand(1, 3)
        if normalization:
            log_prob_mx = add_sample_dimension(mx.nd, mx.nd.array(log_prob, dtype='float64'))
        else:
            p = np.exp(log_prob)
            p = p/p.sum(-1)[:, None]
            log_prob_mx = add_sample_dimension(mx.nd, mx.nd.array(np.log(p), dtype='float64'))
        cate_dist = CategoricalRuntime(log_prob=log_prob_mx, num_classes=num_classes, normalization=normalization, one_hot_encoding=one_hot_encoding)
        samples = cate_dist.draw_samples(num_samples).asnumpy()
        if not one_hot_encoding:
            one_hot = np.zeros((samples.shape[0], num_classes))
            one_hot[np.arange(samples.shape[0]), samples.astype(np.int)[:, 0]] = 1
            samples = one_hot
        rtol, atol = 1e-1, 1e-1
        assert np.allclose(samples.mean(0), cate_dist.mean.asnumpy(), rtol=rtol, atol=atol)
        assert np.allclose(samples.var(0), cate_dist.variance.asnumpy(), rtol=rtol, atol=atol)
