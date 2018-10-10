import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import add_sample_dimension, is_sampled_array, get_num_samples
from mxfusion.components.distributions import Gamma
from mxfusion.util.testutils import numpy_array_reshape
from mxfusion.util.testutils import MockMXNetRandomGenerator


@pytest.mark.usefixtures("set_seed")
class TestGammaDistribution(object):

    @pytest.mark.parametrize("dtype, mean, mean_isSamples, var, var_isSamples, rv, rv_isSamples, num_samples", [
        (np.float64, np.random.rand(5,2), True, np.random.rand(2)+0.1, False, np.random.uniform(0,10,size=(5,3,2)), True, 5),
        (np.float64, np.random.rand(5,2), True, np.random.rand(2)+0.1, False, np.random.uniform(0,10,size=(3,2)), False, 5),
        (np.float64, np.random.rand(2), False, np.random.rand(2)+0.1, False, np.random.uniform(0,10,size=(3,2)), False, 5),
        (np.float64, np.random.rand(5,2), True, np.random.rand(5,3,2)+0.1, True, np.random.uniform(0,10,size=(5,3,2)), True, 5),
        (np.float32, np.random.rand(5,2), True, np.random.rand(2)+0.1, False, np.random.uniform(0,10,size=(5,3,2)), True, 5),
        ])
    def test_log_pdf(self, dtype, mean, mean_isSamples, var, var_isSamples,
                     rv, rv_isSamples, num_samples):
        import scipy as sp


        isSamples_any = any([mean_isSamples, var_isSamples, rv_isSamples])
        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape
        n_dim = 1 + len(rv.shape) if isSamples_any and not rv_isSamples else len(rv.shape)
        mean_np = numpy_array_reshape(mean, mean_isSamples, n_dim)
        var_np = numpy_array_reshape(var, var_isSamples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_isSamples, n_dim)
        log_pdf_np = sp.stats.gamma.logpdf(rv_np.asnumpy(), a=alpha) * (beta)

        gamma = Gamma.define_variable(shape=rv_shape, dtype=dtype).factor
        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_isSamples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        var_mx = mx.nd.array(var, dtype=dtype)
        if not var_isSamples:
            var_mx = add_sample_dimension(mx.nd, var_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_isSamples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {gamma.a.uuid: mean_mx, gamma.b.uuid: var_mx, gamma.random_variable.uuid: rv_mx}
        log_pdf_rt = gamma.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert is_sampled_array(mx.nd, log_pdf_rt) == isSamples_any
        if isSamples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)


    @pytest.mark.parametrize(
        "dtype, mean, mean_isSamples, var, var_isSamples, rv_shape, num_samples",[
        (np.float64, np.random.rand(5,2), True, np.random.rand(2)+0.1, False, (3,2), 5),
        (np.float64, np.random.rand(2), False, np.random.rand(5,2)+0.1, True, (3,2), 5),
        (np.float64, np.random.rand(2), False, np.random.rand(2)+0.1, False, (3,2), 5),
        (np.float64, np.random.rand(5,2), True, np.random.rand(5,3,2)+0.1, True, (3,2), 5),
        (np.float32, np.random.rand(5,2), True, np.random.rand(2)+0.1, False, (3,2), 5),
        ])
    def test_draw_samples(self, dtype, mean, mean_isSamples, var,
                          var_isSamples, rv_shape, num_samples):
        n_dim = 1 + len(rv_shape)
        mean_np = numpy_array_reshape(mean, mean_isSamples, n_dim)
        var_np = numpy_array_reshape(var, var_isSamples, n_dim)

        rand = np.random.randn(num_samples, *rv_shape)
        rv_samples_np = mean_np + rand * np.sqrt(var_np)

        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        gamma = Gamma.define_variable(shape=rv_shape, dtype=dtype,
                                        rand_gen=rand_gen).factor
        mean_mx = mx.nd.array(mean, dtype=dtype)
        if not mean_isSamples:
            mean_mx = add_sample_dimension(mx.nd, mean_mx)
        var_mx = mx.nd.array(var, dtype=dtype)
        if not var_isSamples:
            var_mx = add_sample_dimension(mx.nd, var_mx)
        variables = {gamma.mean.uuid: mean_mx, gamma.variance.uuid: var_mx}
        rv_samples_rt = gamma.draw_samples(
            F=mx.nd, variables=variables, num_samples=num_samples)

        assert np.issubdtype(rv_samples_rt.dtype, dtype)
        assert is_sampled_array(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples

        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(rv_samples_np, rv_samples_rt.asnumpy(), rtol=rtol, atol=atol)
