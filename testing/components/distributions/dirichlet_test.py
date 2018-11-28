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

import numpy as np
import mxnet as mx
from scipy.stats import dirichlet as scipy_dirichlet

from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.components.variables import Variable
from mxfusion.components.distributions import Dirichlet
from mxfusion.util.testutils import numpy_array_reshape
from mxfusion.util.testutils import MockMXNetRandomGenerator


@pytest.mark.usefixtures("set_seed")
class TestDirichletDistribution(object):
    # scipy implementation of dirichlet throws an error if x_i does not sum to one and float32 is
    # not precise enough to have sum close enough to 1 after normalisation so using float64
    @pytest.mark.parametrize("dtype, a, a_is_samples, rv, rv_is_samples, num_samples", [
        (np.float64, np.random.rand(3,2), False, np.random.rand(5, 3, 2), True, 5),
        (np.float64, np.random.rand(10,3,2), True, np.random.rand(10, 3, 2), True, 10),
        (np.float64, np.random.rand(3,2), False, np.random.rand(3, 2), False, 5)
        ])
    def test_log_pdf_with_broadcast(self, dtype, a, a_is_samples, rv, rv_is_samples, num_samples):
        # Add sample dimension if varaible is not samples
        a_mx = mx.nd.array(a, dtype=dtype)
        if not a_is_samples:
            a_mx = add_sample_dimension(mx.nd, a_mx)
        a = a_mx.asnumpy()

        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        rv = rv_mx.asnumpy()

        is_samples_any = a_is_samples or rv_is_samples
        rv_shape = rv.shape[1:]

        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
        a_np = np.broadcast_to(a, (num_samples, 3, 2))
        rv_np = numpy_array_reshape(rv, is_samples_any, n_dim)

        # Initialize rand_gen
        rand = np.random.rand(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        # Calculate correct Dirichlet logpdf
        r = []
        for s in range(len(rv_np)):
            a = []
            for i in range(len(rv_np[s])):
                a.append(scipy_dirichlet.logpdf(rv_np[s][i]/sum(rv_np[s][i]), a_np[s][i]))
            r.append(a)
        log_pdf_np = np.array(r)

        dirichlet = Dirichlet.define_variable(alpha=Variable(), shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        variables = {dirichlet.alpha.uuid: a_mx, dirichlet.random_variable.uuid: rv_mx}
        log_pdf_rt = dirichlet.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == is_samples_any
        if is_samples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples, (get_num_samples(mx.nd, log_pdf_rt), num_samples)
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())

    @pytest.mark.parametrize("dtype, a, a_is_samples, rv, rv_is_samples, num_samples", [
        (np.float64, np.random.rand(5, 3, 2), True, np.random.rand(5, 3, 2), True, 5),
        (np.float64, np.random.rand(10, 3, 2), True, np.random.rand(10, 3, 2), True, 10),
        ])
    def test_log_pdf_no_broadcast(self, dtype, a, a_is_samples,
                                  rv, rv_is_samples, num_samples):

        a_mx = mx.nd.array(a, dtype=dtype)
        if not a_is_samples:
            a_mx = add_sample_dimension(mx.nd, a_mx)
        a = a_mx.asnumpy()

        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        rv = rv_mx.asnumpy()

        is_samples_any = any([a_is_samples, rv_is_samples])
        rv_shape = rv.shape[1:]

        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
        a_np = numpy_array_reshape(a, is_samples_any, n_dim)
        rv_np = numpy_array_reshape(rv, is_samples_any, n_dim)

        rand = np.random.rand(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        r = []
        for s in range(len(rv_np)):
            a = []
            for i in range(len(rv_np[s])):
                a.append(scipy_dirichlet.logpdf(rv_np[s][i]/sum(rv_np[s][i]), a_np[s][i]))
            r.append(a)
        log_pdf_np = np.array(r)

        dirichlet = Dirichlet.define_variable(alpha=Variable(), shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        variables = {dirichlet.alpha.uuid: a_mx, dirichlet.random_variable.uuid: rv_mx}
        log_pdf_rt = dirichlet.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == is_samples_any
        if is_samples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples, (get_num_samples(mx.nd, log_pdf_rt), num_samples)
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())

    @pytest.mark.parametrize("dtype, a, a_is_samples, rv_shape, num_samples", [
        (np.float64, np.random.rand(3,2), False, (3, 2), 5)
        ])
    def test_draw_samples_with_broadcast(self, dtype, a, a_is_samples, rv_shape, num_samples):
        a_mx = mx.nd.array(a, dtype=dtype)
        if not a_is_samples:
            a_mx = add_sample_dimension(mx.nd, a_mx)

        rand = np.random.gamma(shape=a, scale=np.ones(a.shape), size=(num_samples,)+rv_shape)
        draw_samples_np = rand / np.sum(rand)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        dirichlet = Dirichlet.define_variable(alpha=Variable(), shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        variables = {dirichlet.alpha.uuid: a_mx}
        draw_samples_rt = dirichlet.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert np.issubdtype(draw_samples_rt.dtype, dtype)
        assert draw_samples_rt.shape == (5,) + rv_shape
        assert np.allclose(draw_samples_np, draw_samples_rt.asnumpy())

    @pytest.mark.parametrize("dtype, a, a_is_samples, rv_shape, num_samples", [
        (np.float64, np.random.rand(5,3, 2), True, (3, 2), 5)
        ])
    def test_draw_samples_with_broadcast_no_numpy_verification(self, dtype, a, a_is_samples, rv_shape, num_samples):
        a_mx = mx.nd.array(a, dtype=dtype)
        if not a_is_samples:
            a_mx = add_sample_dimension(mx.nd, a_mx)

        dirichlet = Dirichlet.define_variable(alpha=Variable(), shape=rv_shape, dtype=dtype).factor
        variables = {dirichlet.alpha.uuid: a_mx}
        draw_samples_rt = dirichlet.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert np.issubdtype(draw_samples_rt.dtype, dtype)
        assert draw_samples_rt.shape == (5,) + rv_shape

    @pytest.mark.parametrize("dtype, a, a_is_samples, rv_shape, num_samples", [
        (np.float64, np.random.rand(5, 3, 2), True, (3, 2), 5)
        ])
    def test_draw_samples_no_broadcast(self, dtype, a, a_is_samples, rv_shape, num_samples):
        a_mx = mx.nd.array(a, dtype=dtype)
        if not a_is_samples:
            a_mx = add_sample_dimension(mx.nd, a_mx)

        rand = np.random.gamma(shape=a, scale=np.ones(a.shape), size=(num_samples,)+rv_shape)
        draw_samples_np = rand / np.sum(rand)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        dirichlet = Dirichlet.define_variable(alpha=Variable(), shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        variables = {dirichlet.alpha.uuid: a_mx}
        draw_samples_rt = dirichlet.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert np.issubdtype(draw_samples_rt.dtype, dtype)
        assert np.allclose(draw_samples_np, draw_samples_rt.asnumpy())
