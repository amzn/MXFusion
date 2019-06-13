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
from mxfusion import Variable
from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.components.distributions import Normal, MultivariateNormal
from mxfusion.util.testutils import numpy_array_reshape, plot_univariate, plot_bivariate
from mxfusion.util.testutils import MockMXNetRandomGenerator


@pytest.mark.usefixtures("set_seed")
class TestNormalDistribution(object):

    @pytest.mark.parametrize("dtype, mean, mean_isSamples, var, var_isSamples, rv, rv_isSamples, num_samples", [
        (np.float64, np.random.rand(5,3,2), True, np.random.rand(3,2)+0.1, False, np.random.rand(5,3,2), True, 5)
        ])
    def test_log_pdf(self, dtype, mean, mean_isSamples, var, var_isSamples,
                     rv, rv_isSamples, num_samples):
        from scipy.stats import norm

        isSamples_any = any([mean_isSamples, var_isSamples, rv_isSamples])
        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape
        n_dim = 1 + len(rv.shape) if isSamples_any and not rv_isSamples else len(rv.shape)
        mean_np = numpy_array_reshape(mean, mean_isSamples, n_dim)
        var_np = numpy_array_reshape(var, var_isSamples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_isSamples, n_dim)
        log_pdf_np = norm.logpdf(rv_np, mean_np, np.sqrt(var_np))
        normal = Normal.define_variable(shape=rv_shape, mean=Variable(), variance=Variable()).factor
        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_isSamples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        var_mx = mx.nd.array(var, dtype=dtype)
        if not var_isSamples:
            var_mx = add_sample_dimension(mx.nd, var_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_isSamples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {normal.mean.uuid: mean_mx, normal.variance.uuid: var_mx, normal.random_variable.uuid: rv_mx}
        log_pdf_rt = normal.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == isSamples_any
        if isSamples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)


def make_symmetric(array):
    original_shape = array.shape
    d3_array = np.reshape(array, (-1,)+array.shape[-2:])
    d3_array = (d3_array[:,:,:,None]*d3_array[:,:,None,:]).sum(-3)+np.eye(2)
    return np.reshape(d3_array, original_shape)

@pytest.mark.usefixtures("set_seed")
class TestMultivariateNormalDistribution(object):


    @pytest.mark.parametrize("dtype, mean, mean_isSamples, var, var_isSamples, rv, rv_isSamples, num_samples", [
        (np.float32, np.random.rand(3,2), False, make_symmetric(np.random.rand(5,3,2,2)+0.1), True, np.random.rand(5,3,2), True, 5)
        ])
    def test_log_pdf(self, dtype, mean, mean_isSamples, var, var_isSamples,
                        rv, rv_isSamples, num_samples):


        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_isSamples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        mean = mean_mx.asnumpy()

        var_mx = mx.nd.array(var, dtype=dtype)
        if not var_isSamples:
            var_mx = add_sample_dimension(mx.nd, var_mx)
        var = var_mx.asnumpy()

        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_isSamples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        rv = rv_mx.asnumpy()

        print(mean_mx.shape, var_mx.shape, rv_mx.shape)

        from scipy.stats import multivariate_normal
        isSamples_any = any([mean_isSamples, var_isSamples, rv_isSamples])
        rv_shape = rv.shape[1:]

        n_dim = 1 + len(rv.shape) if isSamples_any and not rv_isSamples else len(rv.shape)
        mean_np = np.broadcast_to(mean,(5,3,2))
        var_np = np.broadcast_to(var,(5,3,2,2))
        rv_np = numpy_array_reshape(rv, isSamples_any, n_dim)

        rand = np.random.rand(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        r = []
        for s in range(len(rv_np)):
            a = []
            for i in range(len(rv_np[s])):
                a.append(multivariate_normal.logpdf(rv_np[s][i], mean_np[s][i], var_np[s][i]))
            r.append(a)
        log_pdf_np = np.array(r)

        normal = MultivariateNormal.define_variable(shape=rv_shape, mean=Variable(), covariance=Variable()).factor
        variables = {normal.mean.uuid: mean_mx, normal.covariance.uuid: var_mx, normal.random_variable.uuid: rv_mx}
        log_pdf_rt = normal.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == isSamples_any
        if isSamples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples, (get_num_samples(mx.nd, log_pdf_rt), num_samples)
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())
