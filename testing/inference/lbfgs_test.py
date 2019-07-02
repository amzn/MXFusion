import unittest

import mxfusion as mf
import mxnet as mx
import numpy as np
from mxfusion.common.config import get_default_dtype
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.inference import GradBasedInference
from mxfusion.inference.batch_loop import BatchInferenceLoopScipy
from mxfusion.inference.map import MAP


class LBFGSTests(unittest.TestCase):
    """
    Test class that tests the MXFusion.utils methods.
    """

    def setUp(self):
        dtype = get_default_dtype()

        m = mf.models.Model()
        m.mean = mf.components.Variable()
        m.var = mf.components.Variable(transformation=PositiveTransformation())
        m.N = mf.components.Variable()
        m.x = mf.components.distributions.Normal.define_variable(mean=m.mean, variance=m.var, shape=(m.N,))
        m.y = mf.components.distributions.Normal.define_variable(mean=m.x, variance=mx.nd.array([1], dtype=dtype), shape=(m.N,))
        self.m = m

    def test_optimization(self):
        """
        Tests that the creation of variables from a base gluon block works correctly.
        """

        dtype = get_default_dtype()
        observed = [self.m.y]
        alg = MAP(model=self.m, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoopScipy())

        infr.initialize(y=mx.nd.array(np.random.rand(10), dtype=dtype))
        original_value = infr.params[self.m.x.factor.mean].copy()
        infr.run(y=mx.nd.array(np.random.rand(10), dtype=dtype), max_iter=10)

        # Test that value changes
        assert infr.params[self.m.x.factor.mean] != original_value

    def test_fixed_variable_remains_unchanged(self):
        N = 10
        dtype = get_default_dtype()
        observed = [self.m.y]

        # First check the parameter varies if it isn't fixed
        alg = MAP(model=self.m, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoopScipy())
        infr.initialize(y=mx.nd.array(np.random.rand(N)))
        infr.run(y=mx.nd.array(np.random.rand(N), dtype=dtype), max_iter=10)
        assert infr.params[self.m.x.factor.mean] != mx.nd.ones(1)

        # Now fix parameter and check it does not change
        alg = MAP(model=self.m, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoopScipy())
        infr.initialize(y=mx.nd.array(np.random.rand(N)))
        infr.params.fix_variable(self.m.x.factor.mean, mx.nd.ones(1))
        infr.run(y=mx.nd.array(np.random.rand(N), dtype=dtype), max_iter=10)
        assert infr.params[self.m.x.factor.mean] == mx.nd.ones(1)
