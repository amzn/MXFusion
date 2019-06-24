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
from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.components.distributions import Categorical
from mxfusion.util.testutils import numpy_array_reshape
from mxfusion.util.testutils import MockMXNetRandomGenerator


@pytest.mark.usefixtures("set_seed")
class TestCategoricalDistribution(object):

    @pytest.mark.parametrize("dtype, log_prob, log_prob_isSamples, rv, rv_isSamples, num_samples, one_hot_encoding, normalization", [
        (np.float64, np.random.rand(5,4,3)+1e-2, True, np.random.randint(0,3,size=(5,4,1)), True, 5, False, True)
        ])
    def test_log_pdf(self, dtype, log_prob, log_prob_isSamples, rv, rv_isSamples, num_samples, one_hot_encoding, normalization):

        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape
        n_dim = 1 + len(rv.shape) if not rv_isSamples else len(rv.shape)
        log_prob_np = numpy_array_reshape(log_prob, log_prob_isSamples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_isSamples, n_dim)
        rv_full_shape = (num_samples,)+rv_shape
        rv_np = np.broadcast_to(rv_np, rv_full_shape)
        log_prob_np = np.broadcast_to(log_prob_np, rv_full_shape[:-1]+(3,))

        if normalization:
            log_pdf_np = np.log(np.exp(log_prob_np)/np.exp(log_prob_np).sum(-1, keepdims=True)).reshape(-1, 3)
        else:
            log_pdf_np = log_prob_np.reshape(-1, 3)
        if one_hot_encoding:
            log_pdf_np = (rv_np.reshape(-1, 3)*log_pdf_np).sum(-1).reshape(rv_np.shape[:-1])
        else:
            bool_idx = np.arange(3)[None,:] == rv_np.reshape(-1,1)
            log_pdf_np = log_pdf_np[bool_idx].reshape(rv_np.shape[:-1])

        cat = Categorical.define_variable(0, num_classes=3, one_hot_encoding=one_hot_encoding, normalization=normalization, shape=rv_shape).factor
        log_prob_mx = mx.nd.array(log_prob, dtype=dtype)
        if not log_prob_isSamples:
            log_prob_mx = add_sample_dimension(mx.nd, log_prob_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_isSamples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {cat.log_prob.uuid: log_prob_mx, cat.random_variable.uuid: rv_mx}
        log_pdf_rt = cat.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())
