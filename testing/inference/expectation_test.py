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


import mxnet as mx
import numpy as np
import pytest
import mxfusion as mf
from mxfusion import Model, Variable
from mxfusion.inference import GradBasedInference, TransferInference, ExpectationScoreFunctionAlgorithm, ExpectationAlgorithm


@pytest.mark.usefixtures("set_seed")
class TestExpectationInference(object):
    """
    Test class that tests the MXFusion.inference.expectation classes.
    """

    def make_model(self):
        class Func(mx.gluon.HybridBlock):
            def hybrid_forward(self, F, v2, v3, v4, v1):
                return - (F.sum(v2 * F.minimum(v4, v1) - v3 * v1))

        m = Model()
        N = 1
        m.v1 = Variable(shape=(N,))
        m.v2 = Variable(shape=(N,))
        m.v3 = Variable(shape=(N,))
        m.v4 = mf.components.distributions.Gamma.define_variable(alpha=mx.nd.array([1]),
                                                                     beta=mx.nd.array([0.1]),
                                                                     shape=(N,))
        v5 = mf.components.functions.MXFusionGluonFunction(Func(), num_outputs=1)
        m.v5 = v5(m.v2, m.v3, m.v4, m.v1)
        return m

    @pytest.mark.parametrize("v2, v3", [
        (mx.nd.random.uniform(1,100) * 2, mx.nd.random.uniform(1,100) * 0.5),
        ])
    def test_inference_basic_run(self, v2, v3):
        # TODO test correctness

        m = self.make_model()
        observed = [m.v2, m.v3]
        target_variables = [m.v5]

        infr = GradBasedInference(
            ExpectationScoreFunctionAlgorithm(m, observed, num_samples=10, target_variables=target_variables))

        infr.run(max_iter=1, v2=v2, v3=v3, verbose=True)

        infr2 = TransferInference(
            ExpectationAlgorithm(m, observed, num_samples=10, target_variables=target_variables), infr_params=infr.params)

        infr2.run(max_iter=1, v2=v2, v3=v3, verbose=True)
