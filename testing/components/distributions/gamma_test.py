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
from scipy.stats import gamma

from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.components.distributions import Gamma, GammaMeanVariance
from mxfusion.util.testutils import numpy_array_reshape, plot_univariate
from mxfusion.util.testutils import MockMXNetRandomGenerator


@pytest.mark.usefixtures("set_seed")
class TestGammaDistribution(object):

    @pytest.mark.parametrize("dtype, mean, mean_isSamples, variance, variance_isSamples, rv, rv_isSamples, num_samples", [
        (np.float64, np.random.uniform(0,10,size=(5,3,2)), True,  np.random.uniform(1,10,size=(3,2)), False, np.random.uniform(1,10,size=(5,3,2)), True, 5)
        ])
    def test_log_pdf_mean_variance(self, dtype, mean, mean_isSamples, variance, variance_isSamples,
                     rv, rv_isSamples, num_samples):
        isSamples_any = any([mean_isSamples, variance_isSamples, rv_isSamples])
        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape
        n_dim = 1 + len(rv.shape) if isSamples_any and not rv_isSamples else len(rv.shape)
        mean_np = numpy_array_reshape(mean, mean_isSamples, n_dim)
        variance_np = numpy_array_reshape(variance, variance_isSamples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_isSamples, n_dim)
        beta_np = mean_np / variance_np
        alpha_np = mean_np * beta_np
        log_pdf_np = gamma.logpdf(rv_np, a=alpha_np, loc=0, scale=1. / beta_np)

        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_isSamples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        variance_mx = mx.nd.array(variance, dtype=dtype)
        if not variance_isSamples:
            variance_mx = add_sample_dimension(mx.nd, variance_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_isSamples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        var = GammaMeanVariance.define_variable(mean=mean_mx, variance=variance_mx, shape=rv_shape).factor
        variables = {var.mean.uuid: mean_mx, var.variance.uuid: variance_mx, var.random_variable.uuid: rv_mx}
        log_pdf_rt = var.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == isSamples_any
        if isSamples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype, alpha, alpha_isSamples, beta, beta_isSamples, rv, rv_isSamples, num_samples", [
        (np.float64, np.random.uniform(0,10,size=(5,3,2)), True,  np.random.uniform(1,10,size=(3,2)), False, np.random.uniform(1,10,size=(5,3,2)), True, 5)
        ])
    def test_log_pdf(self, dtype, alpha, alpha_isSamples, beta, beta_isSamples,
                     rv, rv_isSamples, num_samples):
        isSamples_any = any([alpha_isSamples, beta_isSamples, rv_isSamples])
        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape
        n_dim = 1 + len(rv.shape) if isSamples_any and not rv_isSamples else len(rv.shape)
        alpha_np = numpy_array_reshape(alpha, alpha_isSamples, n_dim)
        beta_np = numpy_array_reshape(beta, beta_isSamples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_isSamples, n_dim)
        log_pdf_np = gamma.logpdf(rv_np, a=alpha_np, loc=0, scale=1. / beta_np)

        var = Gamma.define_variable(shape=rv_shape).factor
        alpha_mx = mx.nd.array(alpha, dtype=dtype)
        if not alpha_isSamples:
            alpha_mx = add_sample_dimension(mx.nd, alpha_mx)
        beta_mx = mx.nd.array(beta, dtype=dtype)
        if not beta_isSamples:
            beta_mx = add_sample_dimension(mx.nd, beta_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_isSamples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {var.alpha.uuid: alpha_mx, var.beta.uuid: beta_mx, var.random_variable.uuid: rv_mx}
        log_pdf_rt = var.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == isSamples_any
        if isSamples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)
