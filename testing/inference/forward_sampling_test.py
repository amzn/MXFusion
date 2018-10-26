import unittest
import numpy as np
import mxnet as mx
import mxnet.gluon.nn as nn
import mxfusion as mf
from mxfusion.inference.forward_sampling import VariationalPosteriorForwardSampling
from mxfusion.components.functions import MXFusionGluonFunction
from mxfusion.common.config import get_default_dtype


class ForwardSamplingTests(unittest.TestCase):
    """
    Test class that tests the MXFusion.utils methods.
    """

    def make_model(self, net):
        dtype = get_default_dtype()
        m = mf.models.Model(verbose=False)
        m.N = mf.components.Variable()
        m.f = MXFusionGluonFunction(net, num_outputs=1)
        m.x = mf.components.Variable(shape=(m.N, 1))
        m.r = m.f(m.x)
        for k, v in m.r.factor.parameters.items():
            if k.endswith('_weight') or k.endswith('_bias'):
                v.set_prior(mf.components.distributions.Normal(mean=mx.nd.array([0], dtype=dtype), variance=mx.nd.array([1e6], dtype=dtype)))
        m.y = mf.components.distributions.Categorical.define_variable(log_prob=m.r, num_classes=2, normalization=True, one_hot_encoding=False, shape=(m.N, 1), dtype=dtype)

        return m

    def make_net(self):
        D = 100
        dtype = get_default_dtype()
        net = nn.HybridSequential(prefix='hybrid0_')
        with net.name_scope():
            net.add(nn.Dense(D, activation="tanh", dtype=dtype))
            net.add(nn.Dense(D, activation="tanh", dtype=dtype))
            net.add(nn.Dense(2, flatten=True, dtype=dtype))
        net.initialize(mx.init.Xavier(magnitude=3))
        return net

    def test_forward_sampling(self):
        dtype = get_default_dtype()
        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1) > 0.5
        x_nd, y_nd = mx.nd.array(y, dtype=dtype), mx.nd.array(x, dtype=dtype)

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
