import warnings
warnings.filterwarnings('ignore')
import numpy as np
import unittest
import horovod.mxnet as hvd

import mxnet as mx

class DistributedSVITest(unittest.TestCase):
    """
        Test class that tests MXFusion MAP Inference distributedly using Horovod.
        Run test with command "horovodrun -np {number_of_processors} -H localhost:4 python -m unittest distributed_svi_test.py"
    """

    hvd.init()
    np.random.seed(0)

    from mxfusion.common import config
    config.DEFAULT_DTYPE = 'float64'

    # def make_model_SVI(self):
    #     from mxfusion.components.distributions import Normal
    #     from mxfusion import Model
    #     from mxfusion.components.functions import MXFusionGluonFunction
    #
    #     dtype = 'float64'
    #
    #     m = Model()
    #     mu = Normal.define_variable(mean=mx.nd.array([0], dtype=dtype),
    #                                   variance=mx.nd.array([100], dtype=dtype), shape=(1,))
    #     mu._uuid = "mu"
    #     s_hat = Normal.define_variable(mean=mx.nd.array([5], dtype=dtype),
    #                                      variance=mx.nd.array([100], dtype=dtype),
    #                                      shape=(1,), dtype=dtype)
    #     s_hat._uuid = "s_hat"
    #     trans_mxnet = mx.gluon.nn.HybridLambda(lambda F, x: F.Activation(x, act_type='softrelu'))
    #
    #     m.trans = MXFusionGluonFunction(trans_mxnet, num_outputs=1, broadcastable=True)
    #     s = m.trans(s_hat)
    #     s._uuid = "s"
    #
    #     Y = Normal.define_variable(mean=mu, variance=s, shape=(100,), dtype=dtype)
    #     Y._uuid = "Y"
    #
    #     m.Y = Y
    #     m.s = s
    #     m.s_hat = s_hat
    #     m.mu = mu
    #
    #     return m

    def make_model_SVI(self):
        from mxfusion.components.distributions import Normal
        from mxfusion import Model, Variable
        from mxfusion.components.functions import MXFusionGluonFunction

        dtype = 'float64'

        m = Model()
        m.mu = Normal.define_variable(mean=mx.nd.array([0], dtype=dtype),
                                      variance=mx.nd.array([100], dtype=dtype), shape=(1,))

        m.s_hat = Normal.define_variable(mean=mx.nd.array([5], dtype=dtype),
                                         variance=mx.nd.array([100], dtype=dtype),
                                         shape=(1,), dtype=dtype)

        trans_mxnet = mx.gluon.nn.HybridLambda(lambda F, x: F.Activation(x, act_type='softrelu'))

        m.trans = MXFusionGluonFunction(trans_mxnet, num_outputs=1, broadcastable=True)
        m.s = m.trans(m.s_hat)
        m.X = Variable()

        m.Y = Normal.define_variable(mean=m.mu, variance=m.s, shape=(m.X,), dtype=dtype)

        return m

    # def make_model_SVI_dist(self):
    #     from mxfusion.components.distributions import Normal
    #     from mxfusion import Model, Variable
    #     from mxfusion.components.functions import MXFusionGluonFunction
    #
    #     dtype = 'float64'
    #
    #     m = Model()
    #     mu = Normal.define_variable(mean=mx.nd.array([0], dtype=dtype),
    #                                   variance=mx.nd.array([100], dtype=dtype), shape=(1,))
    #     mu._uuid = "mu"
    #     s_hat = Normal.define_variable(mean=mx.nd.array([5], dtype=dtype),
    #                                      variance=mx.nd.array([100], dtype=dtype),
    #                                      shape=(1,), dtype=dtype)
    #     s_hat._uuid = "s_hat"
    #     trans_mxnet = mx.gluon.nn.HybridLambda(lambda F, x: F.Activation(x, act_type='softrelu'))
    #
    #     m.trans = MXFusionGluonFunction(trans_mxnet, num_outputs=1, broadcastable=True)
    #     s = m.trans(s_hat)
    #     s._uuid = "s"
    #
    #     X = Variable()
    #     X._uuid = "X"
    #
    #     Y = Normal.define_variable(mean=mu, variance=s, shape=(X,), dtype=dtype)
    #     Y._uuid = "Y"
    #
    #     m.Y = Y
    #     m.s = s
    #     m.s_hat = s_hat
    #     m.mu = mu
    #
    #
    #     return m

    def make_model_SVI_dist(self):
        from mxfusion.components.distributions import Normal
        from mxfusion import Model, Variable
        from mxfusion.components.functions import MXFusionGluonFunction

        dtype = 'float64'

        m = Model()
        m.mu = Normal.define_variable(mean=mx.nd.array([0], dtype=dtype),
                                      variance=mx.nd.array([100], dtype=dtype), shape=(1,))
        m.s_hat = Normal.define_variable(mean=mx.nd.array([5], dtype=dtype),
                                         variance=mx.nd.array([100], dtype=dtype),
                                         shape=(1,), dtype=dtype)
        trans_mxnet = mx.gluon.nn.HybridLambda(lambda F, x: F.Activation(x, act_type='softrelu'))

        m.trans = MXFusionGluonFunction(trans_mxnet, num_outputs=1, broadcastable=True)
        m.s = m.trans(m.s_hat)

        m.X = Variable()

        m.Y = Normal.define_variable(mean=m.mu, variance=m.s, shape=(m.X,), dtype=dtype)

        return m



    def test_SVI(self):
        model_single = self.make_model_SVI()
        model_multi = self.make_model_SVI_dist()

        from mxfusion.inference import create_Gaussian_meanfield

        posterior_single = create_Gaussian_meanfield(model=model_single, observed=[model_single.Y])
        posterior_multi = create_Gaussian_meanfield(model=model_multi, observed=[model_multi.Y])

        mean_groundtruth = 3.
        variance_groundtruth = 5.

        dtype = 'float64'

        from mxfusion.inference import GradBasedInference, StochasticVariationalInference, DistributedGradBasedInference

        data = np.random.randn(100) * np.sqrt(variance_groundtruth) + mean_groundtruth

        infr_single = GradBasedInference(inference_algorithm=StochasticVariationalInference(
            model=model_single, posterior=posterior_single, num_samples=10, observed=[model_single.Y]))
        infr_single.run(Y=mx.nd.array(data, dtype=dtype), learning_rate=0.1, verbose=True, max_iter=2000)

        infr_multi = DistributedGradBasedInference(inference_algorithm=StochasticVariationalInference(
            model=model_multi, posterior=posterior_multi, num_samples=10, observed=[model_multi.Y]))
        infr_multi.run(Y=mx.nd.array(data, dtype=dtype), learning_rate=0.1, verbose=True, max_iter=2000)

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

        rtol, atol = 1e-1, 1e-1

        print("mean", mu_mean_single, mu_mean_multi)
        print("mean_std", mu_std_single, mu_std_multi)
        print("s_hat_mean", s_hat_mean_single, s_hat_mean_multi)
        print("s_hat_std", s_hat_std_single, s_hat_std_multi)
        print("s_15", s_15_single, s_15_multi)
        print("s_50", s_50_single, s_50_multi)
        print("s_75", s_85_single, s_85_multi)

        assert np.allclose(mu_mean_single, mu_mean_multi, rtol=rtol, atol=atol)
        assert np.allclose(mu_std_single, mu_std_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_15_single, s_15_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_50_single, s_50_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_85_single, s_85_multi, rtol=rtol, atol=atol)

    def test_SVI_batchloop(self):
        model_single = self.make_model_SVI()
        model_multi = self.make_model_SVI_dist()

        from mxfusion.inference import create_Gaussian_meanfield

        posterior_single = create_Gaussian_meanfield(model=model_single, observed=[model_single.Y])
        posterior_multi = create_Gaussian_meanfield(model=model_multi, observed=[model_multi.Y])

        mean_groundtruth = 3.
        variance_groundtruth = 5.

        dtype = 'float64'

        from mxfusion.inference import GradBasedInference, StochasticVariationalInference, MinibatchInferenceLoop, DistributedGradBasedInference, DistributedMinibatchInferenceLoop

        data = np.random.randn(100) * np.sqrt(variance_groundtruth) + mean_groundtruth

        infr_single = GradBasedInference(inference_algorithm=StochasticVariationalInference(
            model=model_single, posterior=posterior_single, num_samples=10, observed=[model_single.Y]), grad_loop=MinibatchInferenceLoop(batch_size=100))
        infr_single.run(Y=mx.nd.array(data, dtype=dtype), learning_rate=0.1, verbose=True, max_iter=2000)

        infr_multi = DistributedGradBasedInference(inference_algorithm=StochasticVariationalInference(
            model=model_multi, posterior=posterior_multi, num_samples=10, observed=[model_multi.Y]), grad_loop=DistributedMinibatchInferenceLoop(batch_size=100))
        infr_multi.run(Y=mx.nd.array(data, dtype=dtype), learning_rate=0.1, verbose=True, max_iter=2000)

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

        rtol, atol = 1e-1, 1e-1

        print("mean", mu_mean_single, mu_mean_multi)
        print("mean_std", mu_std_single, mu_std_multi)
        print("s_hat_mean", s_hat_mean_single, s_hat_mean_multi)
        print("s_hat_std", s_hat_std_single, s_hat_std_multi)
        print("s_15", s_15_single, s_15_multi)
        print("s_50", s_50_single, s_50_multi)
        print("s_75", s_85_single, s_85_multi)

        assert np.allclose(mu_mean_single, mu_mean_multi, rtol=rtol, atol=atol)
        assert np.allclose(mu_std_single, mu_std_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_15_single, s_15_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_50_single, s_50_multi, rtol=rtol, atol=atol)
        assert np.allclose(s_85_single, s_85_multi, rtol=rtol, atol=atol)





