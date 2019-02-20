import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.components.distributions import Uniform
from mxfusion.util.testutils import numpy_array_reshape, plot_univariate
from mxfusion.util.testutils import MockMXNetRandomGenerator
from scipy.stats import uniform


@pytest.mark.usefixtures("set_seed")
class TestUniformDistribution(object):

    @pytest.mark.parametrize(
        "dtype, low, low_is_samples, high, high_is_samples, rv, rv_is_samples, num_samples", [
            (np.float64, np.random.rand(5,3, 2), True, np.random.rand(3,2) + 1, False, np.random.rand(5, 3, 2) + 0.5, True, 5),
            (np.float64, np.random.rand(5,3, 2), True, np.random.rand(3,2) + 2, False, np.random.rand(3, 2) + 1, False, 5),
            (np.float64, np.random.rand(3,2), False, np.random.rand(3,2) + 2, False, np.random.rand(3, 2) + 1, False, 5),
            (np.float64, np.random.rand(5,3, 2), True, np.random.rand(5, 3, 2) + 2, True, np.random.rand(5, 3, 2) + 1, True,
             5),
            (np.float32, np.random.rand(5,3, 2), True, np.random.rand(3,2) + 2, False, np.random.rand(5, 3, 2) + 1, True, 5),
        ])
    def test_log_pdf(self, dtype, low, low_is_samples, high, high_is_samples, rv, rv_is_samples,
                     num_samples):
        is_samples_any = any([low_is_samples, high_is_samples, rv_is_samples])
        rv_shape = rv.shape[1:] if rv_is_samples else rv.shape
        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
        low_np = numpy_array_reshape(low, low_is_samples, n_dim)
        high_np = numpy_array_reshape(high, high_is_samples, n_dim)
        scale_np = high_np - low_np
        rv_np = numpy_array_reshape(rv, rv_is_samples, n_dim)

        # Note uniform.logpdf takes loc and scale, where loc=a and scale=b-a
        log_pdf_np = uniform.logpdf(rv_np, low_np, scale_np)
        var = Uniform.define_variable(shape=rv_shape, dtype=dtype).factor

        low_mx = mx.nd.array(low, dtype=dtype)
        if not low_is_samples:
            low_mx = add_sample_dimension(mx.nd, low_mx)
        high_mx = mx.nd.array(high, dtype=dtype)
        if not high_is_samples:
            high_mx = add_sample_dimension(mx.nd, high_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {var.low.uuid: low_mx, var.high.uuid: high_mx, var.random_variable.uuid: rv_mx}
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
        "dtype, low, low_is_samples, high, high_is_samples, rv_shape, num_samples", [
            (np.float64, np.random.rand(5, 2), True, np.random.rand(3,2) + 0.1, False, (3, 2), 5),
            (np.float64, np.random.rand(3,2), False, np.random.rand(5, 3,2) + 0.1, True, (3, 2), 5),
            (np.float64, np.random.rand(3,2), False, np.random.rand(3,2) + 0.1, False, (3, 2), 5),
            (np.float64, np.random.rand(5, 3,2), True, np.random.rand(5, 3, 2) + 0.1, True, (3, 2), 5),
            (np.float32, np.random.rand(5,3, 2), True, np.random.rand(3,2) + 0.1, False, (3, 2), 5),
        ])
    def test_draw_samples(self, dtype, low, low_is_samples, high,
                          high_is_samples, rv_shape, num_samples):
        n_dim = 1 + len(rv_shape)
        low_np = numpy_array_reshape(low, low_is_samples, n_dim)
        high_np = numpy_array_reshape(high, high_is_samples, n_dim)

        rv_samples_np = np.random.uniform(low=low_np, high=high_np, size=(num_samples,) + rv_shape)

        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rv_samples_np.flatten(), dtype=dtype))

        var = Uniform.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        low_mx = mx.nd.array(low, dtype=dtype)
        if not low_is_samples:
            low_mx = add_sample_dimension(mx.nd, low_mx)
        high_mx = mx.nd.array(high, dtype=dtype)
        if not high_is_samples:
            high_mx = add_sample_dimension(mx.nd, high_mx)
        variables = {var.low.uuid: low_mx, var.high.uuid: high_mx}

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
        num_samples = 1000

        low = np.array([0.5])
        high = np.array([2])

        rv_shape = (1,)

        low_mx = mx.nd.array(low, dtype=dtype)
        high_mx = mx.nd.array(high, dtype=dtype)

        rand_gen = None
        var = Uniform.define_variable(shape=rv_shape, rand_gen=rand_gen, dtype=dtype).factor
        variables = {var.low.uuid: low_mx, var.high.uuid: high_mx}
        rv_samples_rt = var.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert array_has_samples(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples
        assert rv_samples_rt.dtype == dtype

        if plot:
            plot_univariate(samples=rv_samples_rt, dist=uniform, loc=low[0], scale=high[0] - low[0], buffer=1)

        location_est, scale_est = uniform.fit(rv_samples_rt.asnumpy().ravel())
        location_tol = 1e-2
        scale_tol = 1e-2
        assert np.abs(low[0] - location_est) < location_tol
        assert np.abs(high[0] - scale_est - location_est) < scale_tol
