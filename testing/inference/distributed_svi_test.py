# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
warnings.filterwarnings('ignore')
import numpy as np
import pytest
hvd = pytest.importorskip("horovod.mxnet")
import mxnet as mx

@pytest.mark.usefixtures("set_seed")
class TestDistributedSVI(object):
    """
        Test class that tests MXFusion MAP Inference distributedly using Horovod.
        Run test with command "horovodrun -np {number_of_processors} -H localhost:4 pytest distributed_svi_test.py".
        If run normally with pytest, the distributed training functionality won't be tested.
    """

    hvd.init()

    from mxfusion.common import config
    config.DEFAULT_DTYPE = 'float32'
    mx.context.Context.default_ctx = mx.gpu(hvd.local_rank()) if mx.test_utils.list_gpus() else mx.cpu()

    def make_model_SVI(self):
        from mxfusion.components.distributions import Normal
        from mxfusion import Model, Variable
        from mxfusion.components.functions import MXFusionGluonFunction
        dtype='float32'
        m = Model()
        m.mu = Normal.define_variable(mean=mx.nd.array([0]),
                                      variance=mx.nd.array([100]), shape=(1,))

        m.s_hat = Normal.define_variable(mean=mx.nd.array([5], dtype=dtype),
                                         variance=mx.nd.array([100], dtype=dtype),
                                         shape=(1,), dtype=dtype)

        trans_mxnet = mx.gluon.nn.HybridLambda(lambda F, x: F.Activation(x, act_type='softrelu'))

        m.trans = MXFusionGluonFunction(trans_mxnet, num_outputs=1, broadcastable=True)
        m.s = m.trans(m.s_hat)
        m.X = Variable()

        m.Y = Normal.define_variable(mean=m.mu, variance=m.s, shape=(m.X,), dtype=dtype)

        return m


    def make_inference_SVI(self, model, data, posterior, num_samples=1000, distributed=False, minibatch=False, num_iter=2000, learning_rate=1e-1, batch_size=100):
        from mxfusion.inference import GradBasedInference, DistributedGradBasedInference, BatchInferenceLoop, MinibatchInferenceLoop, DistributedBatchInferenceLoop, DistributedMinibatchInferenceLoop, StochasticVariationalInference

        if distributed:
            infr = DistributedGradBasedInference(inference_algorithm=StochasticVariationalInference(model=model, posterior=posterior, num_samples=num_samples, observed=[model.Y]), grad_loop=DistributedMinibatchInferenceLoop(batch_size=batch_size)) if minibatch else DistributedGradBasedInference(inference_algorithm=StochasticVariationalInference(model=model, posterior=posterior, num_samples=num_samples, observed=[model.Y]), grad_loop=DistributedBatchInferenceLoop())
        else:
            infr = GradBasedInference(inference_algorithm=StochasticVariationalInference(model=model, posterior=posterior,num_samples=num_samples, observed=[model.Y]), grad_loop=MinibatchInferenceLoop(batch_size=batch_size)) if minibatch else GradBasedInference(inference_algorithm=StochasticVariationalInference(model=model, posterior=posterior,num_samples=num_samples, observed=[model.Y]), grad_loop=BatchInferenceLoop())

        infr.run(Y=mx.nd.array(data, dtype='float32'), learning_rate=learning_rate, verbose=True, max_iter=num_iter)

        return infr

    @pytest.mark.parametrize("mean_groundtruth, variance_groundtruth, N, num_samples, max_iter, learning_rate", [
        (5, 3, 200, 10, 50, 1e-1),
        ])
    def test_SVI(self, mean_groundtruth, variance_groundtruth, N, num_samples, max_iter, learning_rate):
        """
            Test the accuracy of distributing training of SVI with comparison to non-distributing training.
            This unit test specifically tests on Batch loop with SVI Inference for gradient optimisation.
            Parameters used for comparisons are mean, standard deviation of mean parameter, mean, standard deviation of variance parameters and 15th, 50th and 85th percentile of variance parameter.
        """

        model_single = self.make_model_SVI()
        model_multi = self.make_model_SVI()

        from mxfusion.inference import create_Gaussian_meanfield

        posterior_single = create_Gaussian_meanfield(model=model_single, observed=[model_single.Y])
        posterior_multi = create_Gaussian_meanfield(model=model_multi, observed=[model_multi.Y])

        data = np.random.randn(N) * np.sqrt(variance_groundtruth) + mean_groundtruth

        infr_single = self.make_inference_SVI(model=model_single, data=data, posterior=posterior_single, num_samples=num_samples, distributed=False, minibatch=False, num_iter=max_iter, learning_rate=learning_rate)
        infr_multi = self.make_inference_SVI(model=model_multi, data=data, posterior=posterior_multi, num_samples=num_samples, distributed=True, minibatch=False, num_iter=max_iter, learning_rate=learning_rate)

        mu_mean_single = infr_single.params[posterior_single.mu.factor.mean].asscalar()
        mu_std_single = np.sqrt(infr_single.params[posterior_single.mu.factor.variance].asscalar())
        s_hat_mean_single = infr_single.params[posterior_single.s_hat.factor.mean].asscalar()
        s_hat_std_single = np.sqrt(infr_single.params[posterior_single.s_hat.factor.variance].asscalar())
        s_15_single = np.log1p(np.exp(s_hat_mean_single - s_hat_std_single))
        s_50_single = np.log1p(np.exp(s_hat_mean_single))
        s_85_single = np.log1p(np.exp(s_hat_mean_single + s_hat_std_single))

        mu_mean_multi = infr_multi.params[posterior_multi.mu.factor.mean].asscalar()
        mu_std_multi = np.sqrt(infr_multi.params[posterior_multi.mu.factor.variance].asscalar())
        s_hat_mean_multi = infr_multi.params[posterior_multi.s_hat.factor.mean].asscalar()
        s_hat_std_multi = np.sqrt(infr_multi.params[posterior_multi.s_hat.factor.variance].asscalar())
        s_15_multi = np.log1p(np.exp(s_hat_mean_multi - s_hat_std_multi))
        s_50_multi = np.log1p(np.exp(s_hat_mean_multi))
        s_85_multi = np.log1p(np.exp(s_hat_mean_multi + s_hat_std_multi))

        if max_iter < 200:
            rtol, atol = 10, 10
        else:
            rtol, atol = 1e-1, 1e-1

        assert np.allclose(mu_mean_single, mu_mean_multi, rtol=rtol, atol=atol)
        assert np.allclose(mu_std_single, mu_std_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_15_single, s_15_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_50_single, s_50_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_85_single, s_85_multi, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("mean_groundtruth, variance_groundtruth, N, num_samples, max_iter, learning_rate, batch_size", [
        (5, 3, 99, 10, 10, 1e-1,
         50),
        ])
    def test_SVI_minibatch(self, mean_groundtruth, variance_groundtruth, N, num_samples, max_iter, learning_rate, batch_size):
        """
            Test the accuracy of distributing training of SVI with comparison to non-distributing training.
            This unit test specifically tests on Minibatch loop with SVI Inference for gradient optimisation.
            Parameters used for comparisons are mean, standard deviation of mean parameter, mean, standard deviation of variance parameters and 15th, 50th and 85th percentile of variance parameter.
        """
        model_single = self.make_model_SVI()
        model_multi = self.make_model_SVI()

        from mxfusion.inference import create_Gaussian_meanfield

        posterior_single = create_Gaussian_meanfield(model=model_single, observed=[model_single.Y])
        posterior_multi = create_Gaussian_meanfield(model=model_multi, observed=[model_multi.Y])

        data = np.random.randn(N) * np.sqrt(variance_groundtruth) + mean_groundtruth

        infr_single = self.make_inference_SVI(model=model_single, data=data, posterior=posterior_single, num_samples=num_samples, distributed=False, minibatch=True, num_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
        infr_multi = self.make_inference_SVI(model=model_multi, data=data, posterior=posterior_multi, num_samples=num_samples, distributed=True, minibatch=True, num_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)

        mu_mean_single = infr_single.params[posterior_single.mu.factor.mean].asscalar()
        mu_std_single = np.sqrt(infr_single.params[posterior_single.mu.factor.variance].asscalar())
        s_hat_mean_single = infr_single.params[posterior_single.s_hat.factor.mean].asscalar()
        s_hat_std_single = np.sqrt(infr_single.params[posterior_single.s_hat.factor.variance].asscalar())
        s_15_single = np.log1p(np.exp(s_hat_mean_single - s_hat_std_single))
        s_50_single = np.log1p(np.exp(s_hat_mean_single))
        s_85_single = np.log1p(np.exp(s_hat_mean_single + s_hat_std_single))

        mu_mean_multi = infr_multi.params[posterior_multi.mu.factor.mean].asscalar()
        mu_std_multi = np.sqrt(infr_multi.params[posterior_multi.mu.factor.variance].asscalar())
        s_hat_mean_multi = infr_multi.params[posterior_multi.s_hat.factor.mean].asscalar()
        s_hat_std_multi = np.sqrt(infr_multi.params[posterior_multi.s_hat.factor.variance].asscalar())
        s_15_multi = np.log1p(np.exp(s_hat_mean_multi - s_hat_std_multi))
        s_50_multi = np.log1p(np.exp(s_hat_mean_multi))
        s_85_multi = np.log1p(np.exp(s_hat_mean_multi + s_hat_std_multi))

        if max_iter < 200:
            rtol, atol = 10, 10
        else:
            rtol, atol = 1e-4, 1e-5

        assert np.allclose(mu_mean_single, mu_mean_multi, rtol=rtol, atol=atol)
        assert np.allclose(mu_std_single, mu_std_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_15_single, s_15_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_50_single, s_50_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_85_single, s_85_multi, rtol=rtol, atol=atol)