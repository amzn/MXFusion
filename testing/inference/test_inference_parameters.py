import unittest

import mxnet as mx
import numpy as np

from mxfusion.common.config import get_default_dtype
from mxfusion.components.variables import PositiveTransformation
from mxfusion.inference import BatchInferenceLoop, GradBasedInference, MAP
import mxfusion as mf


class InferenceParameterTests(unittest.TestCase):
    def setUp(self):
        dtype = get_default_dtype()
        m = mf.models.Model()
        m.mean = mf.components.Variable()
        m.var = mf.components.Variable(transformation=PositiveTransformation())
        m.N = mf.components.Variable()
        m.x = mf.components.distributions.Normal.define_variable(mean=m.mean, variance=m.var, shape=(m.N,))
        m.y = mf.components.distributions.Normal.define_variable(mean=m.x, variance=mx.nd.array([1], dtype=dtype), shape=(m.N,))
        self.m = m

    def test_variable_fixing(self):
        N = 10
        dtype = get_default_dtype()
        observed = [self.m.y]

        # First check the parameter varies if it isn't fixed
        alg = MAP(model=self.m, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
        infr.initialize(y=mx.nd.array(np.random.rand(N)))
        infr.run(y=mx.nd.array(np.random.rand(N), dtype=dtype), max_iter=10)
        assert infr.params[self.m.x.factor.mean] != mx.nd.ones(1)

        # Now fix parameter and check it does not change
        alg = MAP(model=self.m, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
        infr.initialize(y=mx.nd.array(np.random.rand(N)))
        infr.params.fix_variable(self.m.x.factor.mean, mx.nd.ones(1))
        infr.run(y=mx.nd.array(np.random.rand(N), dtype=dtype), max_iter=10)
        assert infr.params[self.m.x.factor.mean] == mx.nd.ones(1)

    def test_variable_unfixing(self):
        N = 10
        y = np.random.rand(N)
        dtype = get_default_dtype()
        observed = [self.m.y]

        # First fix variable and run inference
        alg = MAP(model=self.m, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
        infr.initialize(y=mx.nd.array(np.random.rand(N)))
        infr.params.fix_variable(self.m.x.factor.mean, mx.nd.ones(1))
        infr.run(y=mx.nd.array(y, dtype=dtype), max_iter=10)

        assert infr.params[self.m.x.factor.mean] == mx.nd.ones(1)

        # Now unfix and run inference again
        infr.params.unfix_variable(self.m.x.factor.mean)
        infr.run(y=mx.nd.array(y, dtype=dtype), max_iter=10)

        assert infr.params[self.m.x.factor.mean] != mx.nd.ones(1)
