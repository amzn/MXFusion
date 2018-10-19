import pytest
import numpy as np
import mxnet as mx
from mxfusion.components.variables.runtime_variable import add_sample_dimension, is_sampled_array, get_num_samples
from mxfusion.components.distributions import Dirichlet
from mxfusion.util.testutils import numpy_array_reshape
from mxfusion.util.testutils import MockMXNetRandomGenerator


def make_symmetric(array):
    original_shape = array.shape
    d3_array = np.reshape(array, (-1,)+array.shape[-2:])
    d3_array = (d3_array[:,:,:,None]*d3_array[:,:,None,:]).sum(-3)+np.eye(2)
    return np.reshape(d3_array, original_shape)


@pytest.mark.usefixtures("set_seed")
class TestDirichletDistribution(object):
    # dtype must be float64 to ensure that sum of x_i is always 1
    @pytest.mark.parametrize("dtype, a, a_is_samples, rv, rv_is_samples, num_samples", [
        (np.float64, np.random.rand(2), False, np.random.rand(5, 3, 2), True, 5),
        (np.float64, np.random.rand(2), False, np.random.rand(10, 3, 2), True, 10),
        (np.float64, np.random.rand(2), False, np.random.rand(3, 2), False, 5)
        ])
    def test_log_pdf_with_broadcast(self, dtype, a, a_is_samples, rv, rv_is_samples, num_samples):
        # Add sample dimension if varaible is not samples
        a_mx = mx.nd.array(a, dtype=dtype)
        if not a_is_samples:
            a_mx = add_sample_dimension(mx.nd, a_mx)
        a = a_mx.asnumpy()
        print('a.shape', a.shape)

        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        rv = rv_mx.asnumpy()
        print('rv.shape', rv.shape)

        is_samples_any = a_is_samples or rv_is_samples
        rv_shape = rv.shape[1:]

        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
        a_np = np.broadcast_to(a, (num_samples, 3, 2))
        rv_np = numpy_array_reshape(rv, is_samples_any, n_dim)

        # Initialize rand_gen
        rand = np.random.rand(num_samples, *rv_shape)
        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))

        # Calculate correct Dirichlet logpdf
        from scipy.stats import dirichlet as scipy_dirichlet
        r = []
        for s in range(len(rv_np)):
            a = []
            for i in range(len(rv_np[s])):
                print('rv_np[s][i].shape', rv_np[s][i].shape)
                print('a_np[s][i].shape', a_np[s][i].shape)
                a.append(scipy_dirichlet.logpdf(rv_np[s][i]/sum(rv_np[s][i]), a_np[s][i]))  # NORMALISING
            r.append(a)
        log_pdf_np = np.array(r)

        dirichlet = Dirichlet.define_variable(a=None, shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        variables = {dirichlet.a.uuid: a_mx, dirichlet.random_variable.uuid: rv_mx}
        log_pdf_rt = dirichlet.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert is_sampled_array(mx.nd, log_pdf_rt) == is_samples_any
        if is_samples_any:
            print('log_pdf_rt.shape', log_pdf_rt.shape)
            print('num_samples', num_samples)
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples, (get_num_samples(mx.nd, log_pdf_rt), num_samples)
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())

    # @pytest.mark.parametrize("dtype, a, a_is_samples, var, var_isSamples, rv, rv_is_samples, num_samples", [
    #     (np.float64, np.random.rand(5,3,2), True, np.random.rand(5,3,2), True, 5),
    #     ])
    # def test_log_pdf_no_broadcast(self, dtype, a, a_is_samples,
    #                               rv, rv_is_samples, num_samples):
    # 
    #     a_mx = mx.nd.array(a, dtype=dtype)
    #     if not a_is_samples:
    #         a_mx = add_sample_dimension(mx.nd, a_mx)
    #     a = a_mx.asnumpy()
    # 
    #     rv_mx = mx.nd.array(rv, dtype=dtype)
    #     if not rv_is_samples:
    #         rv_mx = add_sample_dimension(mx.nd, rv_mx)
    #     rv = rv_mx.asnumpy()
    # 
    #     from scipy.stats import dirichlet as scipy_dirichlet
    #     is_samples_any = any([a_is_samples, rv_is_samples])
    #     rv_shape = rv.shape[1:]
    # 
    #     n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
    #     a_np = numpy_array_reshape(a, is_samples_any, n_dim)
    #     rv_np = numpy_array_reshape(rv, is_samples_any, n_dim)
    # 
    #     rand = np.random.rand(num_samples, *rv_shape)
    #     rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))
    # 
    #     r = []
    #     for s in range(len(rv_np)):
    #         a = []
    #         for i in range(len(rv_np[s])):
    #             a.append(scipy_dirichlet.logpdf(rv_np[s][i], a_np[s][i], var_np[s][i]))
    #         r.append(a)
    #     log_pdf_np = np.array(r)
    # 
    #     dirichlet = Dirichlet.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
    #     variables = {dirichlet.a.uuid: a_mx, dirichlet.random_variable.uuid: rv_mx}
    #     log_pdf_rt = dirichlet.log_pdf(F=mx.nd, variables=variables)
    # 
    #     assert np.issubdtype(log_pdf_rt.dtype, dtype)
    #     assert is_sampled_array(mx.nd, log_pdf_rt) == is_samples_any
    #     if is_samples_any:
    #         assert get_num_samples(mx.nd, log_pdf_rt) == num_samples, (get_num_samples(mx.nd, log_pdf_rt), num_samples)
    #     assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy())
    # 
    # @pytest.mark.parametrize(
    #     "dtype, a, a_is_samples, var, var_isSamples, rv_shape, num_samples",[
    #     (np.float64, np.random.rand(2), False, make_symmetric(np.random.rand(2,2)+0.1), False, (5,3,2), 5),
    #     ])
    # def test_draw_samples_with_broadcast(self, dtype, a, a_is_samples, var,
    #                          var_isSamples, rv_shape, num_samples):
    # 
    #     a_mx = mx.nd.array(a, dtype=dtype)
    #     if not a_is_samples:
    #         a_mx = add_sample_dimension(mx.nd, a_mx)
    #     var_mx = mx.nd.array(var, dtype=dtype)
    #     if not var_isSamples:
    #         var_mx = add_sample_dimension(mx.nd, var_mx)
    #     var = var_mx.asnumpy()
    # 
    #     is_samples_any = any([a_is_samples, var_isSamples])
    #     rand = np.random.rand(num_samples, *rv_shape)
    #     rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))
    #     rv_samples_np = a + np.matmul(np.linalg.cholesky(var), np.expand_dims(rand, axis=-1)).sum(-1)
    # 
    #     normal = Dirichlet.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
    #     variables = {normal.a.uuid: a_mx, normal.covariance.uuid: var_mx}
    #     draw_samples_rt = normal.draw_samples(F=mx.nd, variables=variables)
    # 
    #     assert np.issubdtype(draw_samples_rt.dtype, dtype)
    #     assert is_sampled_array(mx.nd, draw_samples_rt) == is_samples_any
    #     if is_samples_any:
    #         assert get_num_samples(mx.nd, draw_samples_rt) == num_samples, (get_num_samples(mx.nd, draw_samples_rt), num_samples)
    # 
    # @pytest.mark.parametrize(
    #     "dtype, a, a_is_samples, var, var_isSamples, rv_shape, num_samples",[
    #     (np.float64, np.random.rand(2), False, make_symmetric(np.random.rand(2,2)+0.1), False, (5,3,2), 5),
    #     (np.float64, np.random.rand(5,2), True, make_symmetric(np.random.rand(2,2)+0.1), False, (5,3,2), 5),
    #     (np.float64, np.random.rand(2), False, make_symmetric(np.random.rand(5,2,2)+0.1), True, (5,3,2), 5),
    #     (np.float64, np.random.rand(5,2), True, make_symmetric(np.random.rand(5,2,2)+0.1), True, (5,3,2), 5),
    #     ])
    # def test_draw_samples_with_broadcast_no_numpy_verification(self, dtype, a, a_is_samples, var,
    #                          var_isSamples, rv_shape, num_samples):
    # 
    #     a_mx = mx.nd.array(a, dtype=dtype)
    #     if not a_is_samples:
    #         a_mx = add_sample_dimension(mx.nd, a_mx)
    #     var_mx = mx.nd.array(var, dtype=dtype)
    #     if not var_isSamples:
    #         var_mx = add_sample_dimension(mx.nd, var_mx)
    #     var = var_mx.asnumpy()
    # 
    #     rand = np.random.rand(num_samples, *rv_shape)
    #     rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))
    # 
    #     normal = Dirichlet.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
    #     variables = {normal.a.uuid: a_mx, normal.covariance.uuid: var_mx}
    #     draw_samples_rt = normal.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)
    # 
    #     assert np.issubdtype(draw_samples_rt.dtype, dtype)
    #     assert is_sampled_array(mx.nd, draw_samples_rt) == True
    # 
    # @pytest.mark.parametrize(
    #     "dtype, a, a_is_samples, var, var_isSamples, rv_shape, num_samples",[
    #     (np.float64, np.random.rand(2), False, make_symmetric(np.random.rand(2,2)+0.1), False, (3,2), 5),
    #     (np.float64, np.random.rand(5,3,2), True, make_symmetric(np.random.rand(5,3,2,2)+0.1), True, (5,3,2), 5),
    #     ])
    # def test_draw_samples_no_broadcast(self, dtype, a, a_is_samples, var,
    #                          var_isSamples, rv_shape, num_samples):
    # 
    #     a_mx = mx.nd.array(a, dtype=dtype)
    #     if not a_is_samples:
    #         a_mx = add_sample_dimension(mx.nd, a_mx)
    #     var_mx = mx.nd.array(var, dtype=dtype)
    #     if not var_isSamples:
    #         var_mx = add_sample_dimension(mx.nd, var_mx)
    #     var = var_mx.asnumpy()
    # 
    #     # n_dim = 1 + len(rv.shape) if is_samples_any else len(rv.shape)
    #     rand = np.random.rand(num_samples, *rv_shape)
    #     rand_gen = MockMXNetRandomGenerator(mx.nd.array(rand.flatten(), dtype=dtype))
    #     rand_exp = np.expand_dims(rand, axis=-1)
    #     lmat = np.linalg.cholesky(var)
    #     temp1 = np.matmul(lmat, rand_exp).sum(-1)
    #     rv_samples_np = a + temp1
    # 
    #     normal = Dirichlet.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
    # 
    #     variables = {normal.a.uuid: a_mx, normal.covariance.uuid: var_mx}
    #     draw_samples_rt = normal.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)
    # 
    #     assert np.issubdtype(draw_samples_rt.dtype, dtype)
    #     assert is_sampled_array(mx.nd, draw_samples_rt) == True
    #     assert get_num_samples(mx.nd, draw_samples_rt) == num_samples, (get_num_samples(mx.nd, draw_samples_rt), num_samples)
