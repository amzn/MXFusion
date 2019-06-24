import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import add_sample_dimension
from mxfusion.components.distributions import PointMass
from mxfusion import Variable


@pytest.mark.usefixtures("set_seed")
class TestPointMassDistribution(object):

    @pytest.mark.parametrize(
        "dtype, location, location_is_samples, rv, rv_is_samples, num_samples", [
        (np.float64, np.random.rand(3,2), False, np.random.rand(3,2), False, 1)
        ])
    def test_log_pdf(self, dtype, location, location_is_samples, rv, rv_is_samples, num_samples):
        is_samples_any = any([location_is_samples, rv_is_samples])
        rv_shape = rv.shape[1:] if rv_is_samples else rv.shape
        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)

        var = PointMass.define_variable(location=Variable(), shape=rv_shape).factor

        location_mx = mx.nd.array(location, dtype=dtype)
        if not location_is_samples:
            location_mx = add_sample_dimension(mx.nd, location_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {var.location.uuid: location_mx, var.random_variable.uuid: rv_mx}
        log_pdf_rt = var.log_pdf(F=mx.nd, variables=variables).asnumpy()

        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(0, log_pdf_rt, rtol=rtol, atol=atol)
