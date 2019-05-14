# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


import unittest
import mxnet as mx
import numpy as np
from mxfusion import Model, Variable
from mxfusion.inference import Inference
from mxfusion.inference.inference_alg import InferenceAlgorithm
from mxfusion.components.distributions import Normal
from mxfusion.components.variables import PositiveTransformation
from mxfusion.inference import GradBasedInference, MAP


class InferenceAlgorithmTests(unittest.TestCase):
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

    def test_change_default_dtype(self):
        from mxfusion.common import config
        config.DEFAULT_DTYPE = 'float64'

        np.random.seed(0)
        mean_groundtruth = 3.
        variance_groundtruth = 5.
        N = 100
        data = np.random.randn(N)*np.sqrt(variance_groundtruth) + mean_groundtruth

        m = Model()
        m.mu = Variable()
        m.s = Variable(transformation=PositiveTransformation())
        m.Y = Normal.define_variable(mean=m.mu, variance=m.s, shape=(100,))

        infr = GradBasedInference(inference_algorithm=MAP(model=m, observed=[m.Y]))
        infr.run(Y=mx.nd.array(data, dtype='float64'), learning_rate=0.1, max_iters=2)

        config.DEFAULT_DTYPE = 'float32'
