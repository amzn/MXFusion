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

from mxfusion.components.variables.runtime_variable import add_sample_dimension, get_num_samples
from mxfusion.components.distributions import SigmoidBernoulli
from mxfusion.util.testutils import numpy_array_reshape


@pytest.mark.usefixtures("set_seed")
class TestBernoulliDistribution(object):

    @pytest.mark.parametrize(
        "dtype, prob_true, prob_true_is_samples, rv, rv_is_samples, num_samples", [
            (np.float64, np.random.beta(a=1, b=1, size=(5, 4, 3)), True, np.random.normal(size=(5, 4, 3)) > 0, True, 5)
        ])
    def test_log_pdf(self, dtype, prob_true, prob_true_is_samples, rv, rv_is_samples, num_samples):

        rv_shape = rv.shape[1:] if rv_is_samples else rv.shape
        n_dim = 1 + len(rv.shape) if not rv_is_samples else len(rv.shape)
        prob_true_np = numpy_array_reshape(prob_true, prob_true_is_samples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_is_samples, n_dim)
        rv_full_shape = (num_samples,)+rv_shape
        rv_np = np.broadcast_to(rv_np, rv_full_shape)

        log_pdf_np = bernoulli.logpmf(k=rv_np, p=1./(1.+np.exp(-prob_true_np)))

        var = SigmoidBernoulli.define_variable(0, shape=rv_shape).factor
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
