import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.components.distributions import PointMass
from mxfusion.util.testutils import numpy_array_reshape, plot_univariate
from mxfusion.util.testutils import MockMXNetRandomGenerator
from mxfusion import Variable
from scipy.stats import laplace


@pytest.mark.usefixtures("set_seed")
class TestPointMassDistribution(object):

    @pytest.mark.parametrize(
        "dtype, location, location_is_samples, rv, rv_is_samples, num_samples", [
        (np.float64, np.random.rand(3,2), False, np.random.rand(3,2), False, 1),
        (np.float32, np.random.rand(5,3,2), True, np.random.rand(5,3,2), True, 5),
        ])
    def test_log_pdf(self, dtype, location, location_is_samples, rv, rv_is_samples, num_samples):
        is_samples_any = any([location_is_samples, rv_is_samples])
        rv_shape = rv.shape[1:] if rv_is_samples else rv.shape
        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)

        var = PointMass.define_variable(location=Variable(), shape=rv_shape, dtype=dtype).factor

        location_mx = mx.nd.array(location, dtype=dtype)
        if not location_is_samples:
            location_mx = add_sample_dimension(mx.nd, location_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {var.location.uuid: location_mx, var.random_variable.uuid: rv_mx}
        log_pdf_rt = var.log_pdf(F=mx.nd, variables=variables)

        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(0, log_pdf_rt, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "dtype, location, location_is_samples, rv_shape, num_samples", [
        (np.float64, np.random.rand(5,3,2), True, (3,2), 5),
        (np.float64, np.random.rand(3,2), False, (3,2), 5),
        ])
    def test_draw_samples(self, dtype, location, location_is_samples, rv_shape, num_samples):
        n_dim = 1 + len(rv_shape)

        var = PointMass.define_variable(location=Variable(), shape=rv_shape, dtype=dtype).factor
        location_mx = mx.nd.array(location, dtype=dtype)
        if not location_is_samples:
            location_mx = add_sample_dimension(mx.nd, location_mx)
        variables = {var.location.uuid: location_mx}

        rv_samples_rt = var.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert np.issubdtype(rv_samples_rt.dtype, dtype)
        assert array_has_samples(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples

        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(location_mx.asnumpy()[0], rv_samples_rt.asnumpy()[0], rtol=rtol, atol=atol)
