import unittest
import mxnet.gluon.nn as nn
import mxnet as mx
import mxfusion.components as mfc
from mxfusion.components.functions import MXFusionGluonFunction
import mxfusion as mf


class FactorTests(unittest.TestCase):
    """
    Tests the MXFusion.core.variable.Factor class.
    """

    def test_replicate_factor_only_self(self):
        m = mf.models.Model(verbose=False)
        m.x = mfc.Variable()
        d = mf.components.distributions.Normal(mean=mx.nd.array([0]), variance=mx.nd.array([1e6]))
        m.x.set_prior(d)
        def func(component):
            return 'recursive', 'recursive'
        x2 = m.x.replicate(replication_function=func)
        self.assertTrue(x2.factor is not None)

    def test_replicate_function_only_self(self):

        self.D = 10
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(nn.Dense(self.D, activation="relu"))

        m = mf.models.Model(verbose=False)
        f = MXFusionGluonFunction(self.net, num_outputs=1)
        m.x = mfc.Variable()
        m.y = f(m.x)

        def func(component):
            return 'recursive', 'recursive'
        y2 = m.y.replicate(replication_function=func)
        self.assertTrue(y2.factor is not None)
