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
from mxfusion.components.variables import Variable
from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples, get_num_samples
from mxfusion.components.distributions.gp.kernels import RBF, Linear, Bias, White, Matern52, Matern32, Matern12
from mxfusion.util.testutils import numpy_array_reshape, prepare_mxnet_array

# These test cases depends on GPy. Put them  in try/except.
try:
    import GPy

    def gpy_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params, num_samples, dtype, mf_kernel_create, gpy_kernel_create):
        X_mx = prepare_mxnet_array(X, X_isSamples, dtype)
        X2_mx = prepare_mxnet_array(X2, X2_isSamples, dtype)
        kern = mf_kernel_create().replicate_self()
        kernel_params_mx = {kern.name + '_' + k:
                            prepare_mxnet_array(v[0], v[1], dtype) for k, v in
                            kernel_params.items()}
        K_XX_mx = kern.K(mx.nd, X=X_mx, **kernel_params_mx)
        K_XX2_mx = kern.K(mx.nd, X=X_mx, X2=X2_mx, **kernel_params_mx)
        Kdiag_mx = kern.Kdiag(mx.nd, X=X_mx, **kernel_params_mx)

        kern_gpy = gpy_kernel_create()
        K_XX_gpy, K_XX2_gpy, Kdiag_gpy = [], [], []
        for i in range(num_samples):
            X_i = X[i] if X_isSamples else X
            X2_i = X2[i] if X2_isSamples else X2
            kernel_params_gpy = {k: v[0][i] if v[1] else v[0] for k, v in
                                 kernel_params.items()}
            for k, v in kernel_params_gpy.items():
                setattr(kern_gpy, k, v)
            K_XX_gpy.append(np.expand_dims(kern_gpy.K(X_i), axis=0))
            K_XX2_gpy.append(np.expand_dims(kern_gpy.K(X_i, X2_i), axis=0))
            Kdiag_gpy.append(np.expand_dims(kern_gpy.Kdiag(X_i), axis=0))
        K_XX_gpy = np.vstack(K_XX_gpy)
        K_XX2_gpy = np.vstack(K_XX2_gpy)
        Kdiag_gpy = np.vstack(Kdiag_gpy)

        assert np.issubdtype(K_XX_mx.dtype, dtype)
        assert np.issubdtype(K_XX2_mx.dtype, dtype)
        assert np.issubdtype(Kdiag_mx.dtype, dtype)
        assert np.allclose(K_XX_gpy, K_XX_mx.asnumpy())
        assert np.allclose(K_XX2_gpy, K_XX2_mx.asnumpy())
        assert np.allclose(Kdiag_gpy, Kdiag_mx.asnumpy())


    def gpy_comb_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params, num_samples, dtype, mf_kernel_create, gpy_kernel_create):
        X_mx = prepare_mxnet_array(X, X_isSamples, dtype)
        X2_mx = prepare_mxnet_array(X2, X2_isSamples, dtype)
        kern = mf_kernel_create().replicate_self()
        kernel_params_mx = {kern.name + '_' + k + '_' + k2:
                            prepare_mxnet_array(v2[0], v2[1], dtype) for k, v in
                            kernel_params.items() for k2, v2 in v.items()}
        K_XX_mx = kern.K(mx.nd, X=X_mx, **kernel_params_mx)
        K_XX2_mx = kern.K(mx.nd, X=X_mx, X2=X2_mx, **kernel_params_mx)
        Kdiag_mx = kern.Kdiag(mx.nd, X=X_mx, **kernel_params_mx)

        kern_gpy = gpy_kernel_create()
        K_XX_gpy, K_XX2_gpy, Kdiag_gpy = [], [], []
        for i in range(num_samples):
            X_i = X[i] if X_isSamples else X
            X2_i = X2[i] if X2_isSamples else X2
            for k, v in kernel_params.items():
                kern_1 = getattr(kern_gpy, k)
                for k2, v2 in v.items():
                    setattr(kern_1, k2, v2[0][i] if v2[1] else v2[0])
            K_XX_gpy.append(np.expand_dims(kern_gpy.K(X_i), axis=0))
            K_XX2_gpy.append(np.expand_dims(kern_gpy.K(X_i, X2_i), axis=0))
            Kdiag_gpy.append(np.expand_dims(kern_gpy.Kdiag(X_i), axis=0))
        K_XX_gpy = np.vstack(K_XX_gpy)
        K_XX2_gpy = np.vstack(K_XX2_gpy)
        Kdiag_gpy = np.vstack(Kdiag_gpy)

        assert np.issubdtype(K_XX_mx.dtype, dtype)
        assert np.issubdtype(K_XX2_mx.dtype, dtype)
        assert np.issubdtype(Kdiag_mx.dtype, dtype)
        assert np.allclose(K_XX_gpy, K_XX_mx.asnumpy())
        assert np.allclose(K_XX2_gpy, K_XX2_mx.asnumpy())
        assert np.allclose(Kdiag_gpy, Kdiag_mx.asnumpy())


    @pytest.mark.usefixtures("set_seed")
    class TestGPKernels(object):

        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples, lengthscale, lengthscale_isSamples, variance, variance_isSamples, num_samples, input_dim, ARD", [
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, 1, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(3,2)+1e-4, True, np.random.rand(1)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(3,1)+1e-4, True, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(1)+1e-4, False, np.random.rand(1)+1e-4, False, 3, 2, False),
            ])
        def test_kernel_as_MXFusionFunction(self, dtype, X, X_isSamples, X2,
            X2_isSamples, lengthscale, lengthscale_isSamples, variance,
            variance_isSamples, num_samples, input_dim, ARD):

            X_mx = prepare_mxnet_array(X, X_isSamples, dtype)
            X2_mx = prepare_mxnet_array(X2, X2_isSamples, dtype)
            var_mx = prepare_mxnet_array(variance, variance_isSamples, dtype)
            l_mx = prepare_mxnet_array(lengthscale, lengthscale_isSamples,
                                       dtype)

            X_mf = Variable(shape=X.shape)
            l_mf = Variable(shape=lengthscale.shape)
            var_mf = Variable(shape=variance.shape)
            rbf = RBF(input_dim, ARD, 1., 1., 'rbf', None, dtype)
            eval = rbf(X_mf, rbf_lengthscale=l_mf, rbf_variance=var_mf).factor
            variables = {eval.X.uuid: X_mx, eval.rbf_lengthscale.uuid: l_mx, eval.rbf_variance.uuid: var_mx}
            res_eval = eval.eval(F=mx.nd, variables=variables)
            kernel_params = rbf.fetch_parameters(variables)
            res_direct = rbf.K(F=mx.nd, X=X_mx, **kernel_params)
            assert np.allclose(res_eval.asnumpy(), res_direct.asnumpy())

            X_mf = Variable(shape=X.shape)
            X2_mf = Variable(shape=X2.shape)
            l_mf = Variable(shape=lengthscale.shape)
            var_mf = Variable(shape=variance.shape)
            rbf = RBF(input_dim, ARD, 1., 1., 'rbf', None, dtype)
            eval = rbf(X_mf, X2_mf, rbf_lengthscale=l_mf, rbf_variance=var_mf).factor
            variables = {eval.X.uuid: X_mx, eval.X2.uuid: X2_mx, eval.rbf_lengthscale.uuid: l_mx, eval.rbf_variance.uuid: var_mx}
            res_eval = eval.eval(F=mx.nd, variables=variables)
            kernel_params = rbf.fetch_parameters(variables)
            res_direct = rbf.K(F=mx.nd, X=X_mx, X2=X2_mx, **kernel_params)
            assert np.allclose(res_eval.asnumpy(), res_direct.asnumpy())

        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples, lengthscale, lengthscale_isSamples, variance, variance_isSamples, num_samples, input_dim, ARD", [
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, 1, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(3,2)+1e-4, True, np.random.rand(1)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(3,1)+1e-4, True, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(1)+1e-4, False, np.random.rand(1)+1e-4, False, 3, 2, False),
            ])
        def test_RBF_kernel(self, dtype, X, X_isSamples, X2, X2_isSamples,
                            lengthscale, lengthscale_isSamples, variance,
                            variance_isSamples, num_samples, input_dim, ARD):
            def create_rbf():
                return RBF(input_dim, ARD, 1., 1., 'rbf', None, dtype)

            def create_gpy_rbf():
                return GPy.kern.RBF(input_dim=input_dim, ARD=ARD)

            kernel_params = {'lengthscale': (lengthscale, lengthscale_isSamples),
                             'variance': (variance, variance_isSamples)}

            gpy_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                            num_samples, dtype, create_rbf, create_gpy_rbf)


        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples,  variances, variances_isSamples, num_samples, input_dim, ARD", [
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, 1, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(3,2)+1e-4, True, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(2)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(1)+1e-4, False, 3, 2, False),
            ])
        def test_Linear_kernel(self, dtype, X, X_isSamples, X2, X2_isSamples,
                               variances, variances_isSamples, num_samples, input_dim,
                               ARD):
            def create_linear():
                return Linear(input_dim, ARD, 1., 'linear', None, dtype)

            def create_gpy_linear():
                return GPy.kern.Linear(input_dim=input_dim, ARD=ARD)

            kernel_params = {'variances': (variances, variances_isSamples)}

            gpy_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                            num_samples, dtype, create_linear, create_gpy_linear)


        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples,  variance, variance_isSamples, num_samples, input_dim", [
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(1)+1e-4, False, 1, 2),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(3,1)+1e-4, True, 3, 2),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(1)+1e-4, False, 3, 2)
            ])
        def test_Bias_kernel(self, dtype, X, X_isSamples, X2, X2_isSamples,
                               variance, variance_isSamples, num_samples, input_dim):
            def create_bias():
                return Bias(input_dim, 1., 'bias', None, dtype)

            def create_gpy_bias():
                return GPy.kern.Bias(input_dim=input_dim)

            kernel_params = {'variance': (variance, variance_isSamples)}

            gpy_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                            num_samples, dtype, create_bias, create_gpy_bias)

        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples,  variance, variance_isSamples, num_samples, input_dim", [
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(1)+1e-4, False, 1, 2),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(3,1)+1e-4, True, 3, 2),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(1)+1e-4, False, 3, 2)
            ])
        def test_White_kernel(self, dtype, X, X_isSamples, X2, X2_isSamples,
                               variance, variance_isSamples, num_samples, input_dim):
            def create_white():
                return White(input_dim, 1., 'bias', None, dtype)

            def create_gpy_white():
                return GPy.kern.White(input_dim=input_dim)

            kernel_params = {'variance': (variance, variance_isSamples)}

            gpy_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                            num_samples, dtype, create_white, create_gpy_white)

        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples,  rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, linear_variances, linear_variances_isSamples, num_samples, input_dim", [
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, np.random.rand(2)+1e-4, False, 1, 2),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, np.random.rand(2)+1e-4, False, 3, 2),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(3,2)+1e-4, True, np.random.rand(1)+1e-4, False, np.random.rand(2)+1e-4, False, 3, 2),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, np.random.rand(3,2)+1e-4, True, 3, 2),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(3,2)+1e-4, True, np.random.rand(3,1)+1e-4, True, np.random.rand(3,2)+1e-4, True, 3, 2)
            ])
        def test_add_kernel(self, dtype, X, X_isSamples, X2, X2_isSamples,  rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, linear_variances, linear_variances_isSamples, num_samples, input_dim):
            def create_rbf_plus_linear():
                return RBF(input_dim, True, 1., 1., 'rbf', None, dtype) + Linear(input_dim, True, 1, 'linear', None, dtype)

            def create_gpy_rbf_plus_linear():
                return GPy.kern.RBF(input_dim=input_dim, ARD=True) + GPy.kern.Linear(input_dim=input_dim, ARD=True)

            kernel_params = {'rbf': {'lengthscale': (rbf_lengthscale, rbf_lengthscale_isSamples), 'variance': (rbf_variance, rbf_variance_isSamples)},
                             'linear': {'variances': (linear_variances, linear_variances_isSamples)}
                             }

            gpy_comb_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                                 num_samples, dtype, create_rbf_plus_linear,
                                 create_gpy_rbf_plus_linear)

        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples,  rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, linear_variances, linear_variances_isSamples, num_samples, input_dim", [
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(3,2)+1e-4, True, np.random.rand(3,1)+1e-4, True, np.random.rand(3,2)+1e-4, True, 3, 2)
            ])
        def test_adding_add_kernel(self, dtype, X, X_isSamples, X2, X2_isSamples,  rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, linear_variances, linear_variances_isSamples, num_samples, input_dim):
            def create_rbf_plus_linear():
                return RBF(input_dim, True, 1., 1., 'rbf', None, dtype) + (RBF(input_dim, True, 1., 1., 'rbf', None, dtype) + Linear(input_dim, True, 1, 'linear', None, dtype))

            def create_gpy_rbf_plus_linear():
                return GPy.kern.RBF(input_dim=input_dim, ARD=True, name='rbf') + GPy.kern.RBF(input_dim=input_dim, ARD=True, name='rbf0') + GPy.kern.Linear(input_dim=input_dim, ARD=True)

            kernel_params = {'rbf': {'lengthscale': (rbf_lengthscale, rbf_lengthscale_isSamples), 'variance': (rbf_variance, rbf_variance_isSamples)},
            'rbf0': {'lengthscale': (rbf_lengthscale, rbf_lengthscale_isSamples), 'variance': (rbf_variance, rbf_variance_isSamples)},
                             'linear': {'variances': (linear_variances, linear_variances_isSamples)}
                             }

            gpy_comb_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                                 num_samples, dtype, create_rbf_plus_linear,
                                 create_gpy_rbf_plus_linear)

        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples,  rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, linear_variances, linear_variances_isSamples, num_samples, input_dim", [
            (np.float64, np.random.rand(5,6), False, np.random.rand(4,6), False, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, np.random.rand(3)+1e-4, False, 1, 6),
            (np.float64, np.random.rand(3,5,6), True, np.random.rand(3,4,6), True, np.random.rand(3,2)+1e-4, True, np.random.rand(3,1)+1e-4, True, np.random.rand(3,3)+1e-4, True, 3, 6)
            ])
        def test_kernel_active_dims(self, dtype, X, X_isSamples, X2, X2_isSamples,  rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, linear_variances, linear_variances_isSamples, num_samples, input_dim):
            def create_rbf_plus_linear():
                return RBF(2, True, 1., 1., 'rbf', [2,3], dtype) + Linear(3, True, 1, 'linear', [4, 1, 5], dtype)

            def create_gpy_rbf_plus_linear():
                return GPy.kern.RBF(input_dim=2, ARD=True, active_dims=[2,3]) + GPy.kern.Linear(input_dim=3, ARD=True, active_dims=[4,1,5])

            kernel_params = {'rbf': {'lengthscale': (rbf_lengthscale, rbf_lengthscale_isSamples), 'variance': (rbf_variance, rbf_variance_isSamples)},
                             'linear': {'variances': (linear_variances, linear_variances_isSamples)}
                             }

            gpy_comb_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                                 num_samples, dtype, create_rbf_plus_linear,
                                 create_gpy_rbf_plus_linear)

        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples,  rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, linear_variances, linear_variances_isSamples, num_samples, input_dim",
            [(np.float64, np.random.rand(5, 2), False, np.random.rand(4, 2), False, np.random.rand(2) + 1e-4, False, np.random.rand(1) + 1e-4, False, np.random.rand(2) + 1e-4, False, 1, 2),
             (np.float64, np.random.rand(3, 5, 2), True, np.random.rand(3, 4, 2), True, np.random.rand(2) + 1e-4, False, np.random.rand(1) + 1e-4, False, np.random.rand(2) + 1e-4, False, 3, 2),
             (np.float64, np.random.rand(5, 2), False, np.random.rand(4, 2), False, np.random.rand(3, 2) + 1e-4, True, np.random.rand(1) + 1e-4, False, np.random.rand(2) + 1e-4, False, 3, 2),
             (np.float64, np.random.rand(5, 2), False, np.random.rand(4, 2), False, np.random.rand(2) + 1e-4, False, np.random.rand(1) + 1e-4, False, np.random.rand(3, 2) + 1e-4, True, 3, 2),
             (np.float64, np.random.rand(3, 5, 2), True, np.random.rand(3, 4, 2), True, np.random.rand(3, 2) + 1e-4, True, np.random.rand(3, 1) + 1e-4, True, np.random.rand(3, 2) + 1e-4, True, 3, 2)])
        def test_mul_kernel(self, dtype, X, X_isSamples, X2, X2_isSamples,  rbf_lengthscale, rbf_lengthscale_isSamples, rbf_variance, rbf_variance_isSamples, linear_variances, linear_variances_isSamples, num_samples, input_dim):
            def create_rbf_plus_linear():
                return RBF(input_dim, True, 1., 1., 'rbf', None, dtype) * Linear(input_dim, True, 1, 'linear', None, dtype)

            def create_gpy_rbf_plus_linear():
                return GPy.kern.RBF(input_dim=input_dim, ARD=True) * GPy.kern.Linear(input_dim=input_dim, ARD=True)

            kernel_params = {'rbf': {'lengthscale': (rbf_lengthscale, rbf_lengthscale_isSamples), 'variance': (rbf_variance, rbf_variance_isSamples)},
                             'linear': {'variances': (linear_variances, linear_variances_isSamples)}
                             }

            gpy_comb_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                                 num_samples, dtype, create_rbf_plus_linear,
                                 create_gpy_rbf_plus_linear)

        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples, lengthscale, lengthscale_isSamples, variance, variance_isSamples, num_samples, input_dim, ARD", [
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, 1, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(3,2)+1e-4, True, np.random.rand(1)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(3,1)+1e-4, True, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(1)+1e-4, False, np.random.rand(1)+1e-4, False, 3, 2, False),
            ])
        def test_Matern52_kernel(self, dtype, X, X_isSamples, X2, X2_isSamples,
                            lengthscale, lengthscale_isSamples, variance,
                            variance_isSamples, num_samples, input_dim, ARD):
            def create_kernel():
                return Matern52(input_dim, ARD, 1., 1., 'rbf', None, dtype)

            def create_gpy_kernel():
                return GPy.kern.Matern52(input_dim=input_dim, ARD=ARD)

            kernel_params = {'lengthscale': (lengthscale, lengthscale_isSamples),
                             'variance': (variance, variance_isSamples)}

            gpy_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                            num_samples, dtype, create_kernel,
                            create_gpy_kernel)

        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples, lengthscale, lengthscale_isSamples, variance, variance_isSamples, num_samples, input_dim, ARD", [
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, 1, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(3,2)+1e-4, True, np.random.rand(1)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(3,1)+1e-4, True, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(1)+1e-4, False, np.random.rand(1)+1e-4, False, 3, 2, False),
            ])
        def test_Matern32_kernel(self, dtype, X, X_isSamples, X2, X2_isSamples,
                            lengthscale, lengthscale_isSamples, variance,
                            variance_isSamples, num_samples, input_dim, ARD):
            def create_kernel():
                return Matern32(input_dim, ARD, 1., 1., 'rbf', None, dtype)

            def create_gpy_kernel():
                return GPy.kern.Matern32(input_dim=input_dim, ARD=ARD)

            kernel_params = {'lengthscale': (lengthscale, lengthscale_isSamples),
                             'variance': (variance, variance_isSamples)}

            gpy_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                            num_samples, dtype, create_kernel,
                            create_gpy_kernel)

        @pytest.mark.parametrize("dtype, X, X_isSamples, X2, X2_isSamples, lengthscale, lengthscale_isSamples, variance, variance_isSamples, num_samples, input_dim, ARD", [
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, 1, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(3,2)+1e-4, True, np.random.rand(1)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(5,2), False, np.random.rand(4,2), False, np.random.rand(2)+1e-4, False, np.random.rand(3,1)+1e-4, True, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(2)+1e-4, False, np.random.rand(1)+1e-4, False, 3, 2, True),
            (np.float64, np.random.rand(3,5,2), True, np.random.rand(3,4,2), True, np.random.rand(1)+1e-4, False, np.random.rand(1)+1e-4, False, 3, 2, False),
            ])
        def test_Matern12_kernel(self, dtype, X, X_isSamples, X2, X2_isSamples,
                            lengthscale, lengthscale_isSamples, variance,
                            variance_isSamples, num_samples, input_dim, ARD):
            def create_kernel():
                return Matern12(input_dim, ARD, 1., 1., 'rbf', None, dtype)

            def create_gpy_kernel():
                return GPy.kern.OU(input_dim=input_dim, ARD=ARD)

            kernel_params = {'lengthscale': (lengthscale, lengthscale_isSamples),
                             'variance': (variance, variance_isSamples)}

            gpy_kernel_test(X, X_isSamples, X2, X2_isSamples, kernel_params,
                            num_samples, dtype, create_kernel,
                            create_gpy_kernel)

except ImportError:
    pass
