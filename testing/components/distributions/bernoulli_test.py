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
from scipy.stats import bernoulli

from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.components.distributions import Bernoulli
from mxfusion.util.testutils import numpy_array_reshape
from mxfusion.util.testutils import MockMXNetRandomGenerator


@pytest.mark.usefixtures("set_seed")
class TestBernoulliDistribution(object):

    @pytest.mark.parametrize(
        "dtype, prob_true, prob_true_is_samples, rv, rv_is_samples, num_samples", [
            (np.float64, np.random.beta(a=1, b=1, size=(5, 4, 3)), True, np.random.normal(size=(5, 4, 3)) > 0, True, 5),
            (np.float64, np.random.beta(a=1, b=1, size=(4, 3)), False, np.random.normal(size=(4, 3)) > 0, False, 1),
            (np.float64, np.random.beta(a=1, b=1, size=(5, 4, 3)), True, np.random.normal(size=(4, 3)) > 0, False, 5),
            (np.float64, np.random.beta(a=1, b=1, size=(4, 3)), False, np.random.normal(size=(5, 4, 3)) > 0, True, 5),
        ])
    def test_log_pdf(self, dtype, prob_true, prob_true_is_samples, rv, rv_is_samples, num_samples):

        rv_shape = rv.shape[1:] if rv_is_samples else rv.shape
        n_dim = 1 + len(rv.shape) if not rv_is_samples else len(rv.shape)
        prob_true_np = numpy_array_reshape(prob_true, prob_true_is_samples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_is_samples, n_dim)
        rv_full_shape = (num_samples,)+rv_shape
        rv_np = np.broadcast_to(rv_np, rv_full_shape)

        log_pdf_np = bernoulli.logpmf(k=rv_np, p=prob_true_np)

        var = Bernoulli.define_variable(0, shape=rv_shape, dtype=dtype).factor
        prob_true_mx = mx.nd.array(prob_true, dtype=dtype)
        if not prob_true_is_samples:
            prob_true_mx = add_sample_dimension(mx.nd, prob_true_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {var.prob_true.uuid: prob_true_mx, var.random_variable.uuid: rv_mx}
        log_pdf_rt = var.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())

    @pytest.mark.parametrize(
        "dtype, prob_true, prob_true_is_samples, rv_shape, num_samples", [
            (np.float64, np.random.rand(5, 4, 3), True, (4, 3), 5),
            (np.float64, np.random.rand(4, 3), False, (4, 3), 5),
            (np.float64, np.random.rand(5, 4, 3), True, (4, 3), 5),
            (np.float64, np.random.rand(4, 3), False, (4, 3), 5),
            (np.float32, np.random.rand(4, 3), False, (4, 3), 5),
        ])
    def test_draw_samples(self, dtype, prob_true, prob_true_is_samples, rv_shape, num_samples):
        rv_full_shape = (num_samples,) + rv_shape

        rand_np = np.random.normal(size=rv_full_shape) > 0
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand_np.flatten(), dtype=dtype))

        rv_samples_np = rand_np

        var = Bernoulli.define_variable(0, shape=rv_shape, rand_gen=rand_gen, dtype=dtype).factor
        prob_true_mx = mx.nd.array(prob_true, dtype=dtype)
        if not prob_true_is_samples:
            prob_true_mx = add_sample_dimension(mx.nd, prob_true_mx)
        variables = {var.prob_true.uuid: prob_true_mx}
        rv_samples_rt = var.draw_samples(
            F=mx.nd, variables=variables, num_samples=num_samples)

        assert array_has_samples(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples
        assert np.array_equal(rv_samples_np, rv_samples_rt.asnumpy().astype(bool))

        # Also make sure the non-mock sampler works
        rand_gen = None
        var = Bernoulli.define_variable(0, shape=rv_shape, rand_gen=rand_gen, dtype=dtype).factor
        prob_true_mx = mx.nd.array(prob_true, dtype=dtype)
        if not prob_true_is_samples:
            prob_true_mx = add_sample_dimension(mx.nd, prob_true_mx)
        variables = {var.prob_true.uuid: prob_true_mx}
        rv_samples_rt = var.draw_samples(
            F=mx.nd, variables=variables, num_samples=num_samples)

        assert array_has_samples(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples
        assert rv_samples_rt.dtype == dtype
