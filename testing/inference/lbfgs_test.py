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

    def test_one_map_example(self):
        """
        Tests that the creation of variables from a base gluon block works correctly.
        """

        dtype = get_default_dtype()
        observed = [self.m.y]
        alg = MAP(model=self.m, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoopScipy())
        infr.run(y=mx.nd.array(np.random.rand(10), dtype=dtype), max_iter=10)
