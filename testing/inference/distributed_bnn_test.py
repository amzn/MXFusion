import warnings
warnings.filterwarnings('ignore')
import numpy as np
import unittest
import horovod.mxnet as hvd
import mxnet.gluon.nn as nn

import mxnet as mx

class DistributedBNNTest(unittest.TestCase):
    """
        Test class that tests MXFusion Bayesian Neuran Network distributedly using Horovod with Stochastic Variational Inference (SVI).
        Run test with command "horovodrun -np {number_of_processors} -H localhost:4 python -m unittest distributed_bnn_test.py"
    """

    hvd.init()
    np.random.seed(0)

    from mxfusion.common import config
    config.DEFAULT_DTYPE = 'float64'

    def make_neural_BNN(self):

        D = 50
        net = nn.HybridSequential(prefix='nn_')
        with net.name_scope():
            net.add(nn.Dense(D, activation="tanh", in_units=1))
            net.add(nn.Dense(D, activation="tanh", in_units=D))
            net.add(nn.Dense(1, flatten=True, in_units=D))
        net.initialize(mx.init.Xavier(magnitude=3))

        return net


    def make_model_BNN(self, net):

        from mxfusion import Variable, Model
        from mxfusion.components.functions import MXFusionGluonFunction
        from mxfusion.components.variables import PositiveTransformation
        from mxfusion.components.distributions import Normal

        m = Model()
        m.N = Variable()
        m.f = MXFusionGluonFunction(net, num_outputs=1, broadcastable=False)
        m.x = Variable(shape=(m.N, 1))
        m.v = Variable(shape=(1,), transformation=PositiveTransformation(), initial_value=mx.nd.array([0.01]))
        m.r = m.f(m.x)

        from mxfusion.components.functions.operators import broadcast_to

        for v in m.r.factor.parameters.values():
            v.set_prior(Normal(mean=broadcast_to(mx.nd.array([0], dtype='float64'), v.shape),
                               variance=broadcast_to(mx.nd.array([1.], dtype='float64'), v.shape)))
        m.y = Normal.define_variable(mean=m.r, variance=broadcast_to(m.v, (m.N, 1)), shape=(m.N, 1))

        return m



    def test_BNN(self):
        import GPy

        N = 1000
        max_iter = 2000
        learning_rate = 1e-2
        net = self.make_neural_BNN()
        np.random.seed(0)
        k = GPy.kern.RBF(1, lengthscale=0.1)
        x = np.random.rand(N, 1)
        y = np.random.multivariate_normal(mean=np.zeros((N,)), cov=k.K(x), size=(1,)).T
        dtype = 'float64'
        model_single = self.make_model_BNN(net)
        model_multi = self.make_model_BNN(net)

        observed_single = [model_single.y, model_single.x]
        observed_multi = [model_multi.y, model_multi.x]

        from mxfusion.inference import create_Gaussian_meanfield, GradBasedInference, DistributedGradBasedInference, StochasticVariationalInference, BatchInferenceLoop, DistributedBatchInferenceLoop

        posterior_single = create_Gaussian_meanfield(model=model_single, observed=observed_single)
        posterior_multi = create_Gaussian_meanfield(model=model_multi, observed=observed_multi)

        alg_single = StochasticVariationalInference(num_samples=3, model=model_single, posterior=posterior_single, observed=observed_single)
        alg_multi = StochasticVariationalInference(num_samples=3, model=model_multi, posterior=posterior_multi, observed=observed_multi)

        infr_single = GradBasedInference(inference_algorithm=alg_single, grad_loop=BatchInferenceLoop())
        infr_multi = DistributedGradBasedInference(inference_algorithm=alg_multi, grad_loop=DistributedBatchInferenceLoop())

        infr_single.initialize(y=(N,1), x=(N,1))
        split_N = int(N/hvd.size())
        infr_multi.initialize(y=(split_N, 1), x=(split_N, 1))

        for v_name, v in model_single.r.factor.parameters.items():
            infr_single.params[posterior_single[v].factor.mean] = net.collect_params()[v_name].data()
            infr_single.params[posterior_single[v].factor.variance] = mx.nd.ones_like(infr_single.params[posterior_single[v].factor.variance]) * 1e-6

        for v_name, v in model_multi.r.factor.parameters.items():
            infr_multi.params[posterior_multi[v].factor.mean] = net.collect_params()[v_name].data()
            infr_multi.params[posterior_multi[v].factor.variance] = mx.nd.ones_like(infr_multi.params[posterior_multi[v].factor.variance]) * 1e-6

        infr_single.run(max_iter=max_iter, learning_rate=learning_rate, y=mx.nd.array(y, dtype=dtype), x=mx.nd.array(x, dtype=dtype), verbose=True)
        infr_multi.run(max_iter=max_iter, learning_rate=learning_rate, y=mx.nd.array(y, dtype=dtype), x=mx.nd.array(x, dtype=dtype), verbose=True)

        from mxfusion.inference import VariationalPosteriorForwardSampling

        xt = np.linspace(0, 1, 100)[:, None]
        infr2_single = VariationalPosteriorForwardSampling(10, [model_single.x], infr_single, [model_single.r])
        infr2_multi = VariationalPosteriorForwardSampling(10, [model_multi.x], infr_multi, [model_multi.r])

        res_single = infr2_single.run(x=mx.nd.array(xt, dtype=dtype))
        yt_single = res_single[0].asnumpy()
        yt_mean_single = yt_single.mean(0)
        yt_std_single = yt_single.std(0)

        res_multi = infr2_multi.run(x=mx.nd.array(xt, dtype=dtype))
        yt_multi = res_multi[0].asnumpy()
        yt_mean_multi = yt_multi.mean(0)
        yt_std_multi= yt_multi.std(0)

        import matplotlib

        rtol, atol = 1e-1, 1e-1

        _, (ax1,ax2) = matplotlib.pyplot.subplots(1,2)

        for i in range(yt_single.shape[0]):
            ax1.plot(xt[:, 0], yt_single[i, :, 0], 'k', alpha=0.2)
        ax1.plot(x[:, 0], y[:, 0], '.')

        for j in range(yt_multi.shape[0]):
            ax2.plot(xt[:, 0], yt_multi[j, :, 0], 'k', alpha=0.2)
        ax2.plot(x[:, 0], y[:, 0], '.')

        ax1.set_title('Single Processor Training on Bayesian Neural Regression')
        ax2.set_title('Multi Processors Training on Bayesian Neural Regression')

        matplotlib.pyplot.show()

        assert np.allclose(yt_mean_single, yt_mean_multi, rtol=rtol, atol=atol)
        assert np.allclose(yt_std_single, yt_std_multi, rtol=rtol, atol=atol)























