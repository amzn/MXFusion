import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import add_sample_dimension, is_sampled_array, get_num_samples
from mxfusion.components.distributions import Gamma
from mxfusion.util.testutils import numpy_array_reshape
from mxfusion.util.testutils import MockMXNetRandomGenerator


@pytest.mark.usefixtures("set_seed")
class TestGammaDistribution(object):

    @pytest.mark.parametrize("dtype, alpha, alpha_isSamples, beta, beta_isSamples, rv, rv_isSamples, num_samples", [
        (np.float64, np.random.uniform(0,10,size=(5,2)), True,  np.random.uniform(1,10,size=(2)), False, np.random.uniform(1,10,size=(5,3,2)), True, 5),
        (np.float64, np.random.uniform(0,10,size=(5,2)), True, np.random.uniform(1,10,size=(2)), False, np.random.uniform(1,10,size=(3,2)), False, 5),
        (np.float64, np.random.uniform(0,10,size=(2)), False, np.random.uniform(1,10,size=(2)), False, np.random.uniform(1,10,size=(3,2)), False, 5),
        (np.float64, np.random.uniform(0,10,size=(5,2)), True, np.random.uniform(1,10,size=(5,3,2)), True, np.random.uniform(1,10,size=(5,3,2)), True, 5),
        (np.float32, np.random.uniform(0,10,size=(5,2)), True, np.random.uniform(1,10,size=(2)), False, np.random.uniform(1,10,size=(5,3,2)), True, 5),
        ])
    def test_log_pdf(self, dtype, alpha, alpha_isSamples, beta, beta_isSamples,
                     rv, rv_isSamples, num_samples):
        import scipy as sp


        isSamples_any = any([alpha_isSamples, beta_isSamples, rv_isSamples])
        rv_shape = rv.shape[1:] if rv_isSamples else rv.shape
        n_dim = 1 + len(rv.shape) if isSamples_any and not rv_isSamples else len(rv.shape)
        alpha_np = numpy_array_reshape(alpha, alpha_isSamples, n_dim)
        beta_np = numpy_array_reshape(beta, beta_isSamples, n_dim)
        rv_np = numpy_array_reshape(rv, rv_isSamples, n_dim)
        log_pdf_np = sp.stats.gamma.logpdf(rv_np, a=alpha_np, loc=0, scale=1./beta_np)

        gamma = Gamma.define_variable(shape=rv_shape, dtype=dtype).factor
        alpha_mx = mx.nd.array(alpha, dtype=dtype)
        if not alpha_isSamples:
            alpha_mx = add_sample_dimension(mx.nd, alpha_mx)
        beta_mx = mx.nd.array(beta, dtype=dtype)
        if not beta_isSamples:
            beta_mx = add_sample_dimension(mx.nd, beta_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_isSamples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {gamma.a.uuid: alpha_mx, gamma.b.uuid: beta_mx, gamma.random_variable.uuid: rv_mx}
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
        "dtype, alpha, alpha_isSamples, beta, beta_isSamples, rv_shape, num_samples",[
        (np.float64, np.random.rand(5,2), True, np.random.rand(2)+0.1, False, (3,2), 5),
        (np.float64, np.random.rand(2), False, np.random.rand(5,2)+0.1, True, (3,2), 5),
        (np.float64, np.random.rand(2), False, np.random.rand(2)+0.1, False, (3,2), 5),
        (np.float64, np.random.rand(5,2), True, np.random.rand(5,3,2)+0.1, True, (3,2), 5),
        (np.float32, np.random.rand(5,2), True, np.random.rand(2)+0.1, False, (3,2), 5),
        ])
    def test_draw_samples(self, dtype, alpha, alpha_isSamples, beta,
                          beta_isSamples, rv_shape, num_samples):
        n_dim = 1 + len(rv_shape)
        out_shape = (num_samples,) + rv_shape
        alpha_np = mx.nd.array(np.broadcast_to(numpy_array_reshape(alpha, alpha_isSamples, n_dim), shape=out_shape), dtype=dtype)
        beta_np = mx.nd.array(np.broadcast_to(numpy_array_reshape(beta, beta_isSamples, n_dim), shape=out_shape), dtype=dtype)

        gamma = Gamma.define_variable(shape=rv_shape, dtype=dtype).factor
        alpha_mx = mx.nd.array(alpha, dtype=dtype)
        if not alpha_isSamples:
            alpha_mx = add_sample_dimension(mx.nd, alpha_mx)
        beta_mx = mx.nd.array(beta, dtype=dtype)
        if not beta_isSamples:
            beta_mx = add_sample_dimension(mx.nd, beta_mx)
        variables = {gamma.a.uuid: alpha_mx, gamma.b.uuid: beta_mx}

        mx.random.seed(0)
        rv_samples_rt = gamma.draw_samples(
            F=mx.nd, variables=variables, num_samples=num_samples)

        mx.random.seed(0)
        rv_samples_mx = mx.nd.random.gamma(alpha=alpha_np, beta=beta_np, dtype=dtype)

        assert np.issubdtype(rv_samples_rt.dtype, dtype)
        assert is_sampled_array(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples

        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(rv_samples_mx.asnumpy(), rv_samples_rt.asnumpy(), rtol=rtol, atol=atol)
