import unittest
import mxnet as mx
import numpy as np
from mxfusion import Model, Variable
from mxfusion.inference import Inference
from mxfusion.inference.inference_alg import InferenceAlgorithm


class InferenceTests(unittest.TestCase):
    """
    Test class that tests the MXFusion.utils methods.
    """

    def test_set_parameters(self):

        class SetValue(InferenceAlgorithm):
            def __init__(self, x, y, model, observed, extra_graphs=None):
                self.x_val = x
                self.y_val = y
                super(SetValue, self).__init__(
                    model=model, observed=observed, extra_graphs=extra_graphs)

            def compute(self, F, variables):
                self.set_parameter(variables, self.model.x, self.x_val)
                self.set_parameter(variables, self.model.y, self.y_val)

        m = Model()
        m.x = Variable(shape=(2,))
        m.y = Variable(shape=(3, 4))

        dtype = 'float64'

        np.random.seed(0)
        x_np = np.random.rand(2)
        y_np = np.random.rand(3, 4)
        x_mx = mx.nd.array(x_np, dtype=dtype)
        y_mx = mx.nd.array(y_np, dtype=dtype)

        infr = Inference(SetValue(x_mx, y_mx, m, []), dtype=dtype)
        infr.run()
        x_res = infr.params[m.x]
        y_res = infr.params[m.y]

        assert np.allclose(x_res.asnumpy(), x_np)
        assert np.allclose(y_res.asnumpy(), y_np)
