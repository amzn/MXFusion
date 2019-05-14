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


import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import add_sample_dimension, array_has_samples


@pytest.mark.usefixtures("set_seed")
class TestFunctionEvaluation(object):

    def _make_test_function_evaluation(self, broadcastable):
        from mxfusion.components.functions.function_evaluation import FunctionEvaluation
        from mxfusion.components import Variable

        class DotFuncEval(FunctionEvaluation):
            def __init__(self):
                inputs = [('A', Variable(shape=(3, 4))), ('B', Variable(shape=(4, 5)))]
                outputs = [('output', Variable(shape=(3, 5)))]
                input_names = ['A', 'B']
                output_names = ['output']
                super(DotFuncEval, self).__init__(inputs=inputs, outputs=outputs,
                                                  input_names=input_names,
                                                  output_names=output_names,
                                                  broadcastable=broadcastable)

            def eval_impl(self, F, A, B):
                return F.linalg.gemm2(A, B)
        return DotFuncEval()

    @pytest.mark.parametrize("dtype, A, A_isSamples, B, B_isSamples, num_samples, broadcastable", [
        (np.float64, np.random.rand(2,3,4), True, np.random.rand(4,5), False, 2, True),
        (np.float64, np.random.rand(2,3,4), True, np.random.rand(2,4,5), True, 2, True),
        (np.float64, np.random.rand(3,4), False, np.random.rand(4,5), False, 0, True),
        (np.float64, np.random.rand(2,3,4), True, np.random.rand(4,5), False, 2, False),
        (np.float64, np.random.rand(2,3,4), True, np.random.rand(2,4,5), True, 2, False),
        (np.float64, np.random.rand(3,4), False, np.random.rand(4,5), False, 0, False)
        ])
    def test_eval(self, dtype, A, A_isSamples, B, B_isSamples, num_samples,
                  broadcastable):

        np_isSamples = A_isSamples or B_isSamples
        if np_isSamples:
            if not A_isSamples:
                A_np = np.expand_dims(A, axis=0)
            else:
                A_np = A
            if not B_isSamples:
                B_np = np.expand_dims(B, axis=0)
            else:
                B_np = B
            res_np = np.einsum('ijk, ikh -> ijh', A_np, B_np)
        else:
            res_np = A.dot(B)

        eval = self._make_test_function_evaluation(broadcastable)
        A_mx = mx.nd.array(A, dtype=dtype)
        if not A_isSamples:
            A_mx = add_sample_dimension(mx.nd, A_mx)
        B_mx = mx.nd.array(B, dtype=dtype)
        if not B_isSamples:
            B_mx = add_sample_dimension(mx.nd, B_mx)
        variables = {eval.A.uuid: A_mx, eval.B.uuid: B_mx}
        res_rt = eval.eval(F=mx.nd, variables=variables)

        assert np_isSamples == array_has_samples(mx.nd, res_rt)
        assert np.allclose(res_np, res_rt.asnumpy())
