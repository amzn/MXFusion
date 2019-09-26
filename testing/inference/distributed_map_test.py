# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pytest
hvd = pytest.importorskip("horovod.mxnet")
import mxnet as mx

@pytest.mark.usefixtures("set_seed")
class TestDistributedMAP(object):
    """
        Test class that tests MXFusion MAP Inference distributedly using Horovod.
        Run test with command "horovodrun -np {number_of_processors} -H localhost:4 pytest -s distributed_map_test.py".
        If run normally with pytest, the distributed training functionality won't be tested.
    """

    hvd.init()
    from mxfusion.common import config
    config.DEFAULT_DTYPE = 'float64'
    mx.context.Context.default_ctx = mx.gpu(hvd.local_rank()) if mx.test_utils.list_gpus() else mx.cpu()

    def make_model_MAP(self):
        from mxfusion.components.distributions import Normal
        from mxfusion.components.variables import PositiveTransformation
        from mxfusion import Variable, Model

        m = Model()
        m.N = Variable()
        m.mu = Variable(initial_value=mx.nd.array([0.1], dtype=np.float64))
        m.s = Variable(transformation=PositiveTransformation(), initial_value=mx.nd.array([1.1], dtype=np.float64))
        m.Y = Normal.define_variable(mean=m.mu, variance=m.s, shape=(m.N,))

        return m

    def make_inference_MAP(self, model, data, distributed=True, minibatch=False, num_iter=2000, learning_rate=1e-1, batch_size=100):
        from mxfusion.inference import GradBasedInference, MAP, DistributedGradBasedInference, BatchInferenceLoop, MinibatchInferenceLoop, DistributedBatchInferenceLoop, DistributedMinibatchInferenceLoop
      
        if distributed:
            infr = DistributedGradBasedInference(inference_algorithm=MAP(model=model, observed=[model.Y]), grad_loop=DistributedMinibatchInferenceLoop(batch_size=batch_size)) if minibatch else DistributedGradBasedInference(inference_algorithm=MAP(model=model, observed=[model.Y]), grad_loop=DistributedBatchInferenceLoop())
        else:
            infr = GradBasedInference(inference_algorithm=MAP(model=model, observed=[model.Y]), grad_loop=MinibatchInferenceLoop(batch_size=batch_size)) if minibatch else GradBasedInference(inference_algorithm=MAP(model=model, observed=[model.Y]), grad_loop=BatchInferenceLoop())

        infr.run(Y=mx.nd.array(data, dtype=np.float64), learning_rate=learning_rate, max_iter=num_iter, verbose=True)

        return infr


    @pytest.mark.parametrize("mean_groundtruth, variance_groundtruth, N, max_iter, learning_rate", [
        (3, 5, 100, 100, 1e-1),
        ])
    def test_MAP(self, mean_groundtruth, variance_groundtruth, N, max_iter, learning_rate):
        """
            Test the accuracy of distributing training of MAP with comparison to non-distributing training.
            This unit test specifically tests on Batch loop with MAP Inference for gradient optimisation.
            Parameters used for comparisons are mean and variance estimated from inferences.
        """

        model_single = self.make_model_MAP()
        model_multi = self.make_model_MAP()
        data = np.random.randn(N) * np.sqrt(variance_groundtruth) + mean_groundtruth

        infr_single = self.make_inference_MAP(model=model_single, data=data, distributed=False, minibatch=False, num_iter=max_iter, learning_rate=learning_rate)
        infr_multi = self.make_inference_MAP(model=model_multi, data=data, distributed=True, minibatch=False, num_iter=max_iter, learning_rate=learning_rate)

        mean_estimated_single = infr_single.params[model_single.mu].asnumpy()
        variance_estimated_single = infr_single.params[model_single.s].asnumpy()

        mean_estimated_double = infr_multi.params[model_multi.mu].asnumpy()
        variance_estimated_double = infr_multi.params[model_multi.s].asnumpy()

        if max_iter < 200:
            rtol, atol = 1, 1
        else:
            rtol, atol = 1e-4, 1e-5

        assert np.allclose(mean_estimated_single, mean_estimated_double, rtol=rtol, atol=atol)
        assert np.allclose(variance_estimated_single, variance_estimated_double, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("mean_groundtruth, variance_groundtruth, N, max_iter, learning_rate, batch_size", [
        (3, 5, 100, 100, 1e-1, 50),
        ])
    def test_MAP_minibatch(self, mean_groundtruth, variance_groundtruth, N, max_iter, learning_rate, batch_size):
        """
            Test the accuracy of distributing training of MAP with comparison to non-distributing training.
            This unit test specifically tests on Minibatch loop with MAP Inference for gradient optimisation.
            Parameters used for comparisons are mean and variance estimated from inferences.
        """

        model_single = self.make_model_MAP()
        model_multi = self.make_model_MAP()
        data = np.random.randn(N) * np.sqrt(variance_groundtruth) + mean_groundtruth

        infr_single = self.make_inference_MAP(model=model_single, data=data, distributed=False, minibatch=True, num_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
        infr_multi = self.make_inference_MAP(model=model_multi, data=data, distributed=True, minibatch=True, num_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)

        mean_estimated_single = infr_single.params[model_single.mu].asnumpy()
        variance_estimated_single = infr_single.params[model_single.s].asnumpy()

        mean_estimated_double = infr_multi.params[model_multi.mu].asnumpy()
        variance_estimated_double = infr_multi.params[model_multi.s].asnumpy()

        if max_iter < 200:
            rtol, atol = 1, 1
        else:
            rtol, atol = 1e-4, 1e-5

        assert np.allclose(mean_estimated_single, mean_estimated_double, rtol=rtol, atol=atol)
        assert np.allclose(variance_estimated_single, variance_estimated_double, rtol=rtol, atol=atol)