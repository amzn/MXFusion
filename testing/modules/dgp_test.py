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


import warnings

import mxnet as mx
import numpy as np
import pytest

from mxfusion.components import Variable
from mxfusion.components.distributions import Normal
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.inference import Inference, MAP, ModulePredictionAlgorithm, TransferInference, create_Gaussian_meanfield, \
    StochasticVariationalInference, GradBasedInference, ForwardSamplingAlgorithm
from mxfusion.models import Model
from mxfusion.modules.gp_modules.deep_gp_regression import DeepGPRegression

warnings.filterwarnings("ignore", category=DeprecationWarning)
dtype = 'float64'


class TestDGPRegressionModule(object):

    def gen_data(self):
        np.random.seed(0)
        D = 1
        X = np.random.rand(10, 3)
        Y = np.random.rand(10, D)
        Z = np.random.rand(3, 3)
        qU_mean = np.random.rand(3, D)
        qU_cov_W = np.random.rand(3, 3)
        qU_cov_diag = np.random.rand(3,)
        noise_var = np.random.rand(1)
        lengthscale = np.random.rand(3)
        variance = np.random.rand(1)
        qU_chol = np.linalg.cholesky(qU_cov_W.dot(qU_cov_W.T) + np.diag(qU_cov_diag))[None, :, :]
        return D, X, Y, Z, noise_var, lengthscale, variance, qU_mean, qU_cov_W, qU_cov_diag, qU_chol

    def gen_mxfusion_model(self, n_dimensions, n_inducing):
        layer_input_dimensions = n_dimensions[:-1]

        m = Model()
        m.N = Variable()

        n_layers = len(layer_input_dimensions)
        m.X = Variable(shape=(m.N, n_dimensions[0]))
        kernels = []
        for i in range(n_layers):
            M = n_inducing[i]
            kernels.append(RBF(layer_input_dimensions[i], lengthscale=0.1, dtype=dtype))
            z_locations = 2 * np.random.rand(M, layer_input_dimensions[i]) - 0.5
            setattr(m, 'Z_' + str(i), Variable(shape=(M, layer_input_dimensions[i]),
                                                        initial_value=mx.nd.array(z_locations)))

        m.noise_var = Variable(transformation=PositiveTransformation())
        inducing_variables = [getattr(m, 'Z_' + str(i)) for i in range(n_layers)]
        m.Y = DeepGPRegression.define_variable(m.X, kernels, m.noise_var, shape=(m.N, n_dimensions[-1]),
                                               inducing_inputs=inducing_variables, n_samples=10, dtype=dtype,
                                               n_outputs=n_dimensions[-1])

        return m

    @pytest.mark.parametrize("input_dim, output_dim, latent_dims", [(2, 1, [3]), (2, 2, [3, 3])])
    def test_prediction(self, input_dim, output_dim, latent_dims):
        n_training_points = 11
        latent_dims = [3]

        X = np.random.rand(n_training_points, input_dim)
        Xt = np.random.rand(n_training_points, input_dim)
        Y = np.random.rand(n_training_points, output_dim)
        n_dimensions = [input_dim] + latent_dims + [output_dim]
        n_inducing = [5, 5]

        m = self.gen_mxfusion_model(n_dimensions, n_inducing)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]), infr_params=infr.params, dtype=np.float64)
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]
        mu_mf, var_mf = res[0].asnumpy()[0], res[1].asnumpy()[0]

        assert mu_mf.shape == (n_training_points, output_dim)
        assert var_mf.shape == (n_training_points, output_dim)

    def test_module_clone(self):
        n_dimensions = [2, 2, 2]
        n_inducing = [5, 5]
        m = self.gen_mxfusion_model(n_dimensions, n_inducing)
        m.clone()
