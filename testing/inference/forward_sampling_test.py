import unittest
import numpy as np
import mxnet as mx
import mxnet.gluon.nn as nn
import mxfusion as mf
from mxfusion.inference.forward_sampling import VariationalPosteriorForwardSampling
from mxfusion.components.functions import MXFusionGluonFunction


class InferenceTests(unittest.TestCase):
    """
    Test class that tests the MXFusion.utils methods.
    """

    def make_model(self, net):
        m = mf.models.Model(verbose=False)
        m.N = mf.components.Variable()
        m.f = MXFusionGluonFunction(net, num_outputs=1)
        m.x = mf.components.Variable(shape=(m.N, 1))
        m.r = m.f(m.x)
        for k, v in m.r.factor.block_variables:
            if k.endswith('_weight') or k.endswith('_bias'):
                v.set_prior(mf.components.distributions.Normal(mean=mx.nd.array([0]), variance=mx.nd.array([1e6])))
        m.y = mf.components.distributions.Categorical.define_variable(log_prob=m.r, num_classes=2, normalization=True, one_hot_encoding=False, shape=(m.N, 1))

        return m

    def make_net(self):
        D = 100
        net = nn.HybridSequential(prefix='hybrid0_')
        with net.name_scope():
            net.add(nn.Dense(D, activation="tanh"))
            net.add(nn.Dense(D, activation="tanh"))
            net.add(nn.Dense(2, flatten=True))
        net.initialize(mx.init.Xavier(magnitude=3))
        return net

    def test_forward_sampling(self):
        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1)>0.5
        x_nd, y_nd = mx.nd.array(y), mx.nd.array(x)

        self.net = self.make_net()
        self.net(x_nd)

        m = self.make_model(self.net)

        from mxfusion.inference.meanfield import create_Gaussian_meanfield
        from mxfusion.inference import StochasticVariationalInference
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference.batch_loop import BatchInferenceLoop
        observed = [m.y, m.x]
        q = create_Gaussian_meanfield(model=m, observed=observed)
        alg = StochasticVariationalInference(num_samples=3, model=m, posterior=q, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
        infr.initialize(y=y_nd, x=x_nd)
        infr._verbose = True
        infr.run(max_iter=2, learning_rate=1e-2, y=y_nd, x=x_nd)

        infr2 = VariationalPosteriorForwardSampling(10, [m.x], infr, [m.r])
        infr2.run(x=x_nd)
