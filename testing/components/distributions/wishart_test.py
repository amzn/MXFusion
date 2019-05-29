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
from scipy.stats import wishart, chi2

from mxfusion.components.distributions import Wishart
from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.util.testutils import MockMXNetRandomGenerator, numpy_array_reshape, plot_univariate, make_spd_matrix


def make_spd_matrices_3d(num_samples, num_dimensions, random_seed):
    matrices = np.zeros((num_samples, num_dimensions, num_dimensions))
    np.random.seed(random_seed)
    for i in range(num_samples):
        matrices[i, :, :] = make_spd_matrix(num_dimensions)
    return matrices


def make_spd_matrices_4d(num_samples, num_data_points, num_dimensions, random_seed):
    matrices = np.zeros((num_samples, num_data_points, num_dimensions, num_dimensions))
    np.random.seed(random_seed)
    for i in range(num_samples):
        for j in range(num_data_points):
            matrices[i, j, :, :] = make_spd_matrix(num_dimensions)
    return matrices


@pytest.mark.usefixtures("set_seed")
class TestWishartDistribution(object):

    @pytest.mark.parametrize("dtype_dof, dtype, degrees_of_freedom, random_seed, scale_is_samples, "
                             "rv_is_samples, num_data_points, num_samples, broadcast",
                             [
                                 (np.int32, np.float32, 2, 0, True, True, 3, 6, False),
                             ])
    def test_log_pdf(self, dtype_dof, dtype, degrees_of_freedom, random_seed,
                     scale_is_samples, rv_is_samples, num_data_points, num_samples, broadcast):
        # Create positive semi-definite matrices
        np.random.seed(random_seed)
        rv = make_spd_matrices_4d(num_samples, num_data_points, degrees_of_freedom, random_seed=random_seed)
        if broadcast:
            scale = make_spd_matrix(dim=degrees_of_freedom)
        else:
            scale = make_spd_matrices_4d(num_samples, num_data_points, degrees_of_freedom, random_seed=random_seed)

        degrees_of_freedom_mx = mx.nd.array([degrees_of_freedom], dtype=dtype_dof)
        degrees_of_freedom = degrees_of_freedom_mx.asnumpy()[0]  # ensures the correct dtype

        scale_mx = mx.nd.array(scale, dtype=dtype)
        if not scale_is_samples:
            scale_mx = add_sample_dimension(mx.nd, scale_mx)
        scale = scale_mx.asnumpy()

        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        rv = rv_mx.asnumpy()

        is_samples_any = scale_is_samples or rv_is_samples

        if broadcast:
            scale_np = np.broadcast_to(scale, rv.shape)
        else:
            n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
            scale_np = numpy_array_reshape(scale, is_samples_any, n_dim)

        rv_np = numpy_array_reshape(rv, is_samples_any, degrees_of_freedom)

        r = []
        for s in range(num_samples):
            a = []
            for i in range(num_data_points):
                a.append(wishart.logpdf(rv_np[s][i], df=degrees_of_freedom, scale=scale_np[s][i]))
            r.append(a)
        log_pdf_np = np.array(r)

        var = Wishart.define_variable(shape=rv.shape[1:], dtype=dtype, rand_gen=None).factor
        variables = {var.degrees_of_freedom.uuid: degrees_of_freedom_mx, var.scale.uuid: scale_mx,
                     var.random_variable.uuid: rv_mx}
        log_pdf_rt = var.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert array_has_samples(mx.nd, log_pdf_rt) == is_samples_any
        if is_samples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples, (get_num_samples(mx.nd, log_pdf_rt), num_samples)
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())

    @pytest.mark.parametrize(
        "dtype_dof, dtype, degrees_of_freedom, scale, scale_is_samples, rv_shape, num_samples", [
            (np.int64, np.float64, 3, make_spd_matrix(3), False, (3, 3), 5),
            (np.int64, np.float64, 3, make_spd_matrices_4d(5, 5, 3, 0), True, (5, 3, 3), 5),
        ])
    def test_draw_samples_no_broadcast(self, dtype_dof, dtype, degrees_of_freedom, scale,
                                       scale_is_samples, rv_shape, num_samples):
        degrees_of_freedom_mx = mx.nd.array([degrees_of_freedom], dtype=dtype_dof)
        scale_mx = mx.nd.array(scale, dtype=dtype)
        if not scale_is_samples:
            scale_mx = add_sample_dimension(mx.nd, scale_mx)

        rand = np.random.rand(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        var = Wishart.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        variables = {var.degrees_of_freedom.uuid: degrees_of_freedom_mx, var.scale.uuid: scale_mx}
        draw_samples_rt = var.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert np.issubdtype(draw_samples_rt.dtype, dtype)
        assert array_has_samples(mx.nd, draw_samples_rt)
        assert get_num_samples(mx.nd, draw_samples_rt) == num_samples, (get_num_samples(mx.nd, draw_samples_rt),
                                                                        num_samples)

    @pytest.mark.usefixtures("set_seed")
    def test_draw_samples_1d(self, plot=False):
        # Also make sure the non-mock sampler works by drawing 1D samples (should collapse to chi^2)
        dtype = np.float32
        dtype_dof = np.int32
        num_samples = 20000

        dof = 3
        scale = np.array([[1]])

        rv_shape = scale.shape

        dof_mx = mx.nd.array([dof], dtype=dtype_dof)
        scale_mx = add_sample_dimension(mx.nd, mx.nd.array(scale, dtype=dtype))

        rand_gen = None
        var = Wishart.define_variable(shape=rv_shape, rand_gen=rand_gen, dtype=dtype).factor
        variables = {var.degrees_of_freedom.uuid: dof_mx, var.scale.uuid: scale_mx}
        rv_samples_rt = var.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert array_has_samples(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples
        assert rv_samples_rt.dtype == dtype

        if plot:
            plot_univariate(samples=rv_samples_rt, dist=chi2, df=dof)

        # Note that the chi-squared fitting doesn't do a great job, so we have a slack tolerance
        dof_est, _, _ = chi2.fit(rv_samples_rt.asnumpy().ravel())
        dof_tol = 1.5

        assert np.abs(dof - dof_est) < dof_tol
