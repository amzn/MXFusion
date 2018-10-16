import pytest
import mxnet as mx
from sklearn.datasets import make_spd_matrix
import numpy as np
from scipy.stats import wishart

from mxfusion.components.distributions import Wishart
from mxfusion.components.variables.runtime_variable import add_sample_dimension, is_sampled_array, get_num_samples
from mxfusion.util.testutils import MockMXNetRandomGenerator, numpy_array_reshape


@pytest.mark.usefixtures("set_seed")
class TestWishartDistribution(object):

    @pytest.mark.parametrize("dtype, degrees_of_freedom, random_state, scale_is_samples, "
                             "rv_is_samples, num_data_points, num_samples",
                             [
                                 (np.float32, 2, 0, True, True, 3, 5),
                             ])
    def test_log_pdf_with_broadcast(self, dtype, degrees_of_freedom, random_state,
                                    scale_is_samples, rv_is_samples, num_data_points, num_samples):
        # Create random variables
        rv = np.zeros((num_samples, num_data_points, degrees_of_freedom, degrees_of_freedom))
        for i in range(num_samples):
            for j in range(num_data_points):
                # TODO: should be PD not PSD?
                rv[i, j, :, :] = make_spd_matrix(degrees_of_freedom)

        # Create a positive semi-definite matrix to act as the scale matrix
        # TODO: should be PD not PSD?
        scale = make_spd_matrix(n_dim=degrees_of_freedom, random_state=random_state)

        degrees_of_freedom_mx = mx.nd.array([degrees_of_freedom], dtype=int)
        # degrees_of_freedom = degrees_of_freedom_mx.asnumpy()

        scale_mx = mx.nd.array(scale, dtype=dtype)
        if not scale_is_samples:
            scale_mx = add_sample_dimension(mx.nd, scale_mx)
        scale = scale_mx.asnumpy()

        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        rv = rv_mx.asnumpy()

        is_samples_any = scale_is_samples or rv_is_samples
        rv_shape = rv.shape[1:]

        degrees_of_freedom = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
        scale_np = np.broadcast_to(scale, rv.shape)
        rv_np = numpy_array_reshape(rv, is_samples_any, degrees_of_freedom)

        rand = np.random.rand(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        r = []
        for s in range(num_samples):
            a = []
            for i in range(num_data_points):
                a.append(wishart.logpdf(rv_np[s][i], df=degrees_of_freedom, scale=scale_np[s][i]))
            r.append(a)
        log_pdf_np = np.array(r)

        var = Wishart.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        variables = {var.degrees_of_freedom.uuid: degrees_of_freedom_mx, var.scale.uuid: scale_mx,
                     var.random_variable.uuid: rv_mx}
        log_pdf_rt = var.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert is_sampled_array(mx.nd, log_pdf_rt) == is_samples_any
        if is_samples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples, (get_num_samples(mx.nd, log_pdf_rt), num_samples)
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())

    # @pytest.mark.parametrize("dtype, n_dim, random_state, scale_is_samples, rv, rv_is_samples, num_samples", [
    #     (np.float32, 2, 0, False, np.random.rand(5, 3, 2), True, 5),
    # ])
    # def test_log_pdf_no_broadcast(self, dtype, n_dim, random_state, scale_is_samples, rv, rv_is_samples, num_samples):
    #     # Create a positive semi-definite matrix to act as the scale matrix
    #     # TODO: should be PD not PSD?
    #     scale = make_spd_matrix(n_dim=n_dim, random_state=random_state)
    #
    #     scale_mx = mx.nd.array(scale, dtype=dtype)
    #     if not scale_is_samples:
    #         scale_mx = add_sample_dimension(mx.nd, scale_mx)
    #     scale = scale_mx.asnumpy()
    #
    #     rv_mx = mx.nd.array(rv, dtype=dtype)
    #     if not rv_is_samples:
    #         rv_mx = add_sample_dimension(mx.nd, rv_mx)
    #     rv = rv_mx.asnumpy()
    #
    #     from scipy.stats import multivariate_normal
    #     is_samples_any = scale_is_samples or rv_is_samples
    #     rv_shape = rv.shape[1:]
    #
    #     n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
    #     mean_np = numpy_array_reshape(mean, is_samples_any, n_dim)
    #     var_np = numpy_array_reshape(scale, is_samples_any, n_dim)
    #     rv_np = numpy_array_reshape(rv, is_samples_any, n_dim)
    #
    #     rand = np.random.rand(num_samples, *rv_shape)
    #     rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))
    #
    #     r = []
    #     for s in range(len(rv_np)):
    #         a = []
    #         for i in range(len(rv_np[s])):
    #             a.append(multivariate_normal.logpdf(rv_np[s][i], mean_np[s][i], var_np[s][i]))
    #         r.append(a)
    #     log_pdf_np = np.array(r)
    #
    #     var = Wishart.define_variable(degrees_of_freedom=n_dim, shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
    #     variables = {var.covariance.uuid: scale_mx, var.random_variable.uuid: rv_mx}
    #     log_pdf_rt = var.log_pdf(F=mx.nd, variables=variables)
    #
    #     assert np.issubdtype(log_pdf_rt.dtype, dtype)
    #     assert is_sampled_array(mx.nd, log_pdf_rt) == is_samples_any
    #     if is_samples_any:
    #         assert get_num_samples(mx.nd, log_pdf_rt) == num_samples, (get_num_samples(mx.nd, log_pdf_rt), num_samples)
    #     assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())

    # @pytest.mark.parametrize(
    #     "dtype, mean, mean_is_samples, var, var_is_samples, rv_shape, num_samples",[
    #     (np.float64, np.random.rand(2), False, make_symmetric(np.random.rand(2,2)+0.1), False, (5,3,2), 5),
    #     ])
    # def test_draw_samples_with_broadcast(self, dtype, mean, mean_is_samples, var,
    #                          var_is_samples, rv_shape, num_samples):
    #
    #     scale_mx = mx.nd.array(var, dtype=dtype)
    #     if not var_is_samples:
    #         scale_mx = add_sample_dimension(mx.nd, scale_mx)
    #     var = scale_mx.asnumpy()
    #
    #     is_samples_any = any([mean_is_samples, var_is_samples])
    #     rand = np.random.rand(num_samples, *rv_shape)
    #     rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))
    #     rv_samples_np = mean + np.matmul(np.linalg.cholesky(var), np.expand_dims(rand, axis=-1)).sum(-1)
    #
    #     var = Wishart.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
    #     variables = {var.scale.uuid: scale_mx}
    #     draw_samples_rt = var.draw_samples(F=mx.nd, variables=variables)
    #
    #     assert np.issubdtype(draw_samples_rt.dtype, dtype)
    #     assert is_sampled_array(mx.nd, draw_samples_rt) == is_samples_any
    #     if is_samples_any:
    #         assert get_num_samples(mx.nd, draw_samples_rt) == num_samples, (get_num_samples(mx.nd, draw_samples_rt), num_samples)
    #
    #
    # @pytest.mark.parametrize(
    #     "dtype, mean, mean_is_samples, var, var_is_samples, rv_shape, num_samples",[
    #     (np.float64, np.random.rand(2), False, make_symmetric(np.random.rand(2,2)+0.1), False, (3,2), 5),
    #     (np.float64, np.random.rand(5,3,2), True, make_symmetric(np.random.rand(5,3,2,2)+0.1), True, (5,3,2), 5),
    #     ])
    # def test_draw_samples_no_broadcast(self, dtype, mean, mean_is_samples, var,
    #                          var_is_samples, rv_shape, num_samples):
    #
    #     mean_mx = mx.nd.array(mean, dtype=dtype)
    #     if not mean_is_samples:
    #         mean_mx = add_sample_dimension(mx.nd, mean_mx)
    #     var_mx = mx.nd.array(var, dtype=dtype)
    #     if not var_is_samples:
    #         var_mx = add_sample_dimension(mx.nd, var_mx)
    #     var = var_mx.asnumpy()
    #     # var = (var[:,:,:,None]*var[:,None,:,:]).sum(-2)+np.eye(2)
    #     # var_mx = mx.nd.array(var, dtype=dtype)
    #
    #     is_samples_any = any([mean_is_samples, var_is_samples])
    #     # n_dim = 1 + len(rv.shape) if is_samples_any else len(rv.shape)
    #     # mean_np = numpy_array_reshape(mean, mean_is_samples, n_dim)
    #     # var_np = numpy_array_reshape(var, var_is_samples, n_dim)
    #     rand = np.random.rand(num_samples, *rv_shape)
    #     rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))
    #     rand_exp = np.expand_dims(rand, axis=-1)
    #     lmat = np.linalg.cholesky(var)
    #     temp1 = np.matmul(lmat, rand_exp).sum(-1)
    #     rv_samples_np = mean + temp1
    #
    #     normal = MultivariateNormal.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
    #
    #     variables = {normal.mean.uuid: mean_mx, normal.covariance.uuid: var_mx}
    #     draw_samples_rt = normal.draw_samples(F=mx.nd, variables=variables)
    #
    #     assert np.issubdtype(draw_samples_rt.dtype, dtype)
    #     assert is_sampled_array(mx.nd, draw_samples_rt) == is_samples_any
    #     if is_samples_any:
    #         assert get_num_samples(mx.nd, draw_samples_rt) == num_samples, (get_num_samples(mx.nd, draw_samples_rt), num_samples)
    #     # assert np.allclose(rv_samples_np, draw_samples_rt.asnumpy()), (rv_samples_np, draw_samples_rt.asnumpy())
