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
from scipy.stats import norm, multivariate_normal

from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.components.distributions import NormalMeanPrecision, MultivariateNormalMeanPrecision
from mxfusion.util.testutils import numpy_array_reshape
from mxfusion.util.testutils import MockMXNetRandomGenerator


@pytest.mark.usefixtures("set_seed")
class TestNormalPrecisionDistribution(object):

    @pytest.mark.parametrize(
        "dtype, mean, mean_is_samples, precision, precision_is_samples, rv, rv_is_samples, num_samples", [
        (np.float64, np.random.rand(5,3,2), True, np.random.rand(3,2)+0.1, False, np.random.rand(5,3,2), True, 5),
        (np.float64, np.random.rand(3,2), False, np.random.rand(5,3,2)+0.1, True, np.random.rand(5,3,2), True, 5),
        (np.float64, np.random.rand(3,2), False, np.random.rand(3,2)+0.1, False, np.random.rand(5,3,2), True, 5),
        (np.float64, np.random.rand(3,2), False, np.random.rand(3,2)+0.1, False, np.random.rand(3,2), False, 1),
        (np.float32, np.random.rand(5,3,2), True, np.random.rand(3,2)+0.1, False, np.random.rand(5,3,2), True, 5),
        ])
    def test_log_pdf(self, dtype, mean, mean_is_samples, precision, precision_is_samples,
                     rv, rv_is_samples, num_samples):
        is_samples_any = any([mean_is_samples, precision_is_samples, rv_is_samples])
        rv_shape = rv.shape[1:] if rv_is_samples else rv.shape
        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
        mean_np = numpy_array_reshape(mean, mean_is_samples, n_dim)
        precision_np = numpy_array_reshape(precision, precision_is_samples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_is_samples, n_dim)
        log_pdf_np = norm.logpdf(rv_np, mean_np, np.power(precision_np, -0.5))

        var = NormalMeanPrecision.define_variable(shape=rv_shape, dtype=dtype).factor
        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_is_samples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        precision_mx = mx.nd.array(precision, dtype=dtype)
        if not precision_is_samples:
            precision_mx = add_sample_dimension(mx.nd, precision_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {var.mean.uuid: mean_mx, var.precision.uuid: precision_mx, var.random_variable.uuid: rv_mx}
        log_pdf_rt = var.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == is_samples_any
        if is_samples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "dtype, mean, mean_is_samples, precision, precision_is_samples, rv_shape, num_samples", [
        (np.float64, np.random.rand(5,3,2), True, np.random.rand(3,2)+0.1, False, (3,2), 5),
        (np.float64, np.random.rand(3,2), False, np.random.rand(5,3,2)+0.1, True, (3,2), 5),
        (np.float64, np.random.rand(3,2), False, np.random.rand(3,2)+0.1, False, (3,2), 5),
        (np.float64, np.random.rand(5,3,2), True, np.random.rand(5,3,2)+0.1, True, (3,2), 5),
        (np.float32, np.random.rand(5,3,2), True, np.random.rand(3,2)+0.1, False, (3,2), 5),
        ])
    def test_draw_samples(self, dtype, mean, mean_is_samples, precision,
                          precision_is_samples, rv_shape, num_samples):
        n_dim = 1 + len(rv_shape)
        mean_np = numpy_array_reshape(mean, mean_is_samples, n_dim)
        precision_np = numpy_array_reshape(precision, precision_is_samples, n_dim)

        rand = np.random.randn(num_samples, *rv_shape)
        rv_samples_np = mean_np + rand * np.power(precision_np, -0.5)

        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        var = NormalMeanPrecision.define_variable(shape=rv_shape, dtype=dtype,
                                                  rand_gen=rand_gen).factor
        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_is_samples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        precision_mx = mx.nd.array(precision, dtype=dtype)
        if not precision_is_samples:
            precision_mx = add_sample_dimension(mx.nd, precision_mx)
        variables = {var.mean.uuid: mean_mx, var.precision.uuid: precision_mx}
        rv_samples_rt = var.draw_samples(
            F=mx.nd, variables=variables, num_samples=num_samples)

        assert np.issubdtype(rv_samples_rt.dtype, dtype)
        assert array_has_samples(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples

        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(rv_samples_np, rv_samples_rt.asnumpy(), rtol=rtol, atol=atol)


def make_symmetric(array):
    original_shape = array.shape
    d3_array = np.reshape(array, (-1,)+array.shape[-2:])
    d3_array = (d3_array[:,:,:,None]*d3_array[:,:,None,:]).sum(-3)+np.eye(2)
    return np.reshape(d3_array, original_shape)


@pytest.mark.usefixtures("set_seed")
class TestMultivariateNormalMeanPrecisionDistribution(object):

    @pytest.mark.parametrize(
        "dtype, mean, mean_is_samples, precision, precision_is_samples, rv, rv_is_samples, num_samples", [
        (np.float32, np.random.rand(3,2), False, make_symmetric(np.random.rand(5,3,2,2)+0.1), True, np.random.rand(5,3,2), True, 5),
        (np.float32, np.random.rand(5,3,2), True, make_symmetric(np.random.rand(3,2,2)+0.1), False, np.random.rand(5,3,2), True, 5),
        (np.float32, np.random.rand(3,2), False, make_symmetric(np.random.rand(3,2,2)+0.1), False, np.random.rand(5,3,2), True, 5),
        ])
    def test_log_pdf_with_broadcast(self, dtype, mean, mean_is_samples, precision, precision_is_samples,
                        rv, rv_is_samples, num_samples):

        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_is_samples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        mean = mean_mx.asnumpy()

        precision_mx = mx.nd.array(precision, dtype=dtype)
        if not precision_is_samples:
            precision_mx = add_sample_dimension(mx.nd, precision_mx)
        precision = precision_mx.asnumpy()

        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        rv = rv_mx.asnumpy()

        is_samples_any = any([mean_is_samples, precision_is_samples, rv_is_samples])
        rv_shape = rv.shape[1:]

        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
        mean_np = np.broadcast_to(mean, (5, 3, 2))
        precision_np = np.broadcast_to(precision,  (5, 3, 2, 2))
        rv_np = numpy_array_reshape(rv, is_samples_any, n_dim)

        rand = np.random.rand(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        r = []
        for s in range(len(rv_np)):
            a = []
            for i in range(len(rv_np[s])):
                a.append(multivariate_normal.logpdf(rv_np[s][i], mean_np[s][i], np.linalg.inv(precision_np[s][i])))
            r.append(a)
        log_pdf_np = np.array(r)

        normal = MultivariateNormalMeanPrecision.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        variables = {
            normal.mean.uuid: mean_mx, normal.precision.uuid: precision_mx, normal.random_variable.uuid: rv_mx}
        log_pdf_rt = normal.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == is_samples_any
        if is_samples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples, (get_num_samples(mx.nd, log_pdf_rt), num_samples)
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())

    @pytest.mark.parametrize(
        "dtype, mean, mean_is_samples, precision, precision_is_samples, rv, rv_is_samples, num_samples", [
            (np.float64, np.random.rand(5, 3, 2), True, make_symmetric(np.random.rand(5, 3, 2, 2) + 0.1), True,
             np.random.rand(5, 3, 2), True, 5),
        ])
    def test_log_pdf_no_broadcast(self, dtype, mean, mean_is_samples, precision, precision_is_samples,
                                  rv, rv_is_samples, num_samples):

        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_is_samples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        mean = mean_mx.asnumpy()

        precision_mx = mx.nd.array(precision, dtype=dtype)
        if not precision_is_samples:
            precision_mx = add_sample_dimension(mx.nd, precision_mx)
        precision = precision_mx.asnumpy()

        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        rv = rv_mx.asnumpy()

        is_samples_any = any([mean_is_samples, precision_is_samples, rv_is_samples])
        rv_shape = rv.shape[1:]

        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
        mean_np = numpy_array_reshape(mean, is_samples_any, n_dim)
        precision_np = numpy_array_reshape(precision, is_samples_any, n_dim)
        rv_np = numpy_array_reshape(rv, is_samples_any, n_dim)

        rand = np.random.rand(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        r = []
        for s in range(len(rv_np)):
            a = []
            for i in range(len(rv_np[s])):
                a.append(multivariate_normal.logpdf(rv_np[s][i], mean_np[s][i], np.linalg.inv(precision_np[s][i])))
            r.append(a)
        log_pdf_np = np.array(r)

        normal = MultivariateNormalMeanPrecision.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        variables = {normal.mean.uuid: mean_mx, normal.precision.uuid: precision_mx, normal.random_variable.uuid: rv_mx}
        log_pdf_rt = normal.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == is_samples_any
        if is_samples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples, (get_num_samples(mx.nd, log_pdf_rt), num_samples)
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())

    @pytest.mark.parametrize(
        "dtype, mean, mean_is_samples, precision, precision_is_samples, rv_shape, num_samples", [
        (np.float64, np.random.rand(3,2), False, make_symmetric(np.random.rand(3,2,2)+0.1), False, (3,2), 5),
        ])
    def test_draw_samples_with_broadcast(self, dtype, mean, mean_is_samples, precision,
                                         precision_is_samples, rv_shape, num_samples):

        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_is_samples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        precision_mx = mx.nd.array(precision, dtype=dtype)
        if not precision_is_samples:
            precision_mx = add_sample_dimension(mx.nd, precision_mx)
        # precision = precision_mx.asnumpy()

        is_samples_any = any([mean_is_samples, precision_is_samples])
        rand = np.random.rand(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))
        # rv_samples_np = mean + np.matmul(np.linalg.cholesky(precision), np.expand_dims(rand, axis=-1)).sum(-1)

        normal = MultivariateNormalMeanPrecision.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        variables = {normal.mean.uuid: mean_mx, normal.precision.uuid: precision_mx}
        draw_samples_rt = normal.draw_samples(F=mx.nd, variables=variables)

        assert np.issubdtype(draw_samples_rt.dtype, dtype)
        assert array_has_samples(mx.nd, draw_samples_rt) == is_samples_any
        if is_samples_any:
            assert get_num_samples(mx.nd, draw_samples_rt) == num_samples, \
                (get_num_samples(mx.nd, draw_samples_rt), num_samples)

    @pytest.mark.parametrize(
        "dtype, mean, mean_is_samples, precision, precision_is_samples, rv_shape, num_samples", [
        (np.float64, np.random.rand(3,2), False, make_symmetric(np.random.rand(3,2,2)+0.1), False, (3,2), 5),
        (np.float64, np.random.rand(3,2), False, make_symmetric(np.random.rand(5,3,2,2)+0.1), True, (3,2), 5),
        (np.float64, np.random.rand(5,3,2), True, make_symmetric(np.random.rand(3,2,2)+0.1), False, (3,2), 5),
        (np.float64, np.random.rand(5,3,2), True, make_symmetric(np.random.rand(5,3,2,2)+0.1), True, (3,2), 5),
        ])
    def test_draw_samples_no_broadcast(self, dtype, mean, mean_is_samples, precision,
                                       precision_is_samples, rv_shape, num_samples):

        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_is_samples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        precision_mx = mx.nd.array(precision, dtype=dtype)
        if not precision_is_samples:
            precision_mx = add_sample_dimension(mx.nd, precision_mx)
        # precision = precision_mx.asnumpy()

        # n_dim = 1 + len(rv.shape) if is_samples_any else len(rv.shape)
        rand = np.random.rand(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))
        # rand_exp = np.expand_dims(rand, axis=-1)
        # lmat = np.linalg.cholesky(precision)
        # temp1 = np.matmul(lmat, rand_exp).sum(-1)
        # rv_samples_np = mean + temp1

        normal = MultivariateNormalMeanPrecision.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor

        variables = {normal.mean.uuid: mean_mx, normal.precision.uuid: precision_mx}
        draw_samples_rt = normal.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert np.issubdtype(draw_samples_rt.dtype, dtype)
        assert array_has_samples(mx.nd, draw_samples_rt)
        assert get_num_samples(mx.nd, draw_samples_rt) == num_samples, \
            (get_num_samples(mx.nd, draw_samples_rt), num_samples)
