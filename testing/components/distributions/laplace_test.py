import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.components.distributions import Laplace
from mxfusion.util.testutils import numpy_array_reshape, plot_univariate
from mxfusion.util.testutils import MockMXNetRandomGenerator
from scipy.stats import laplace


@pytest.mark.usefixtures("set_seed")
class TestLaplaceDistribution(object):

    @pytest.mark.parametrize(
        "dtype, location, location_is_samples, scale, scale_is_samples, rv, rv_is_samples, num_samples", [
        (np.float64, np.random.rand(5,3,2), True, np.random.rand(3,2)+0.1, False, np.random.rand(5,3,2), True, 5),
        (np.float64, np.random.rand(3,2), False, np.random.rand(5,3,2)+0.1, True, np.random.rand(5,3,2), True, 5),
        (np.float64, np.random.rand(3,2), False, np.random.rand(3,2)+0.1, False, np.random.rand(5,3,2), True, 5),
        (np.float64, np.random.rand(3,2), False, np.random.rand(3,2)+0.1, False, np.random.rand(3,2), False, 1),
        (np.float32, np.random.rand(5,3,2), True, np.random.rand(3,2)+0.1, False, np.random.rand(5,3,2), True, 5),
        ])
    def test_log_pdf(self, dtype, location, location_is_samples, scale, scale_is_samples, rv, rv_is_samples,
                     num_samples):
        is_samples_any = any([location_is_samples, scale_is_samples, rv_is_samples])
        rv_shape = rv.shape[1:] if rv_is_samples else rv.shape
        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
        location_np = numpy_array_reshape(location, location_is_samples, n_dim)
        scale_np = numpy_array_reshape(scale, scale_is_samples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_is_samples, n_dim)

        log_pdf_np = laplace.logpdf(rv_np, location_np, scale_np)
        var = Laplace.define_variable(shape=rv_shape, dtype=dtype).factor

        location_mx = mx.nd.array(location, dtype=dtype)
        if not location_is_samples:
            location_mx = add_sample_dimension(mx.nd, location_mx)
        var_mx = mx.nd.array(scale, dtype=dtype)
        if not scale_is_samples:
            var_mx = add_sample_dimension(mx.nd, var_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {var.location.uuid: location_mx, var.scale.uuid: var_mx, var.random_variable.uuid: rv_mx}
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
        "dtype, location, location_is_samples, scale, scale_is_samples, rv_shape, num_samples", [
        (np.float64, np.random.rand(5,3,2), True, np.random.rand(3,2)+0.1, False, (3,2), 5),
        (np.float64, np.random.rand(3,2), False, np.random.rand(5,3,2)+0.1, True, (3,2), 5),
        (np.float64, np.random.rand(3,2), False, np.random.rand(3,2)+0.1, False, (3,2), 5),
        (np.float64, np.random.rand(5,3,2), True, np.random.rand(5,3,2)+0.1, True, (3,2), 5),
        (np.float32, np.random.rand(5,3,2), True, np.random.rand(3,2)+0.1, False, (3,2), 5),
        ])
    def test_draw_samples(self, dtype, location, location_is_samples, scale,
                          scale_is_samples, rv_shape, num_samples):
        n_dim = 1 + len(rv_shape)
        location_np = numpy_array_reshape(location, location_is_samples, n_dim)
        scale_np = numpy_array_reshape(scale, scale_is_samples, n_dim)

        rand = np.random.laplace(size=(num_samples,) + rv_shape)
        rv_samples_np = location_np + rand * scale_np

        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        var = Laplace.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        location_mx = mx.nd.array(location, dtype=dtype)
        if not location_is_samples:
            location_mx = add_sample_dimension(mx.nd, location_mx)
        scale_mx = mx.nd.array(scale, dtype=dtype)
        if not scale_is_samples:
            scale_mx = add_sample_dimension(mx.nd, scale_mx)
        variables = {var.location.uuid: location_mx, var.scale.uuid: scale_mx}

        rv_samples_rt = var.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert np.issubdtype(rv_samples_rt.dtype, dtype)
        assert array_has_samples(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples

        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(rv_samples_np, rv_samples_rt.asnumpy(), rtol=rtol, atol=atol)

    def test_draw_samples_non_mock(self, plot=False):
        # Also make sure the non-mock sampler works
        dtype = np.float32
        num_samples = 100000

        location = np.array([0.5])
        scale = np.array([2])

        rv_shape = (1,)

        location_mx = add_sample_dimension(mx.nd, mx.nd.array(location, dtype=dtype))
        scale_mx = add_sample_dimension(mx.nd, mx.nd.array(scale, dtype=dtype))

        rand_gen = None
        var = Laplace.define_variable(shape=rv_shape, rand_gen=rand_gen, dtype=dtype).factor
        variables = {var.location.uuid: location_mx, var.scale.uuid: scale_mx}
        rv_samples_rt = var.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert array_has_samples(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples
        assert rv_samples_rt.dtype == dtype

        if plot:
            plot_univariate(samples=rv_samples_rt, dist=laplace, loc=location[0], scale=scale[0])

        location_est, scale_est = laplace.fit(rv_samples_rt.asnumpy().ravel())
        location_tol = 1e-2
        scale_tol = 1e-2
        assert np.abs(location[0] - location_est) < location_tol
        assert np.abs(scale[0] - scale_est) < scale_tol
