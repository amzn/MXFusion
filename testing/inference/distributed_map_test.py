import warnings
warnings.filterwarnings('ignore')
import numpy as np
import unittest
import horovod.mxnet as hvd

import mxnet as mx


class DistributedMAPTest(unittest.TestCase):
    """
    Test class that tests MXFusion MAP Inference distributedly using Horovod.
    Run test with command "horovodrun -np {number_of_processors} -H localhost:4 python -m unittest distributed_map_test.py"
    """

    hvd.init()
    np.random.seed(0)

    from mxfusion.common import config
    config.DEFAULT_DTYPE = 'float64'

    def make_model_MAP(self):
        from mxfusion.components.distributions import Normal
        from mxfusion.components.variables import PositiveTransformation
        from mxfusion import Variable, Model

        m = Model()
        m.mu = Variable()
        m.s = Variable(transformation=PositiveTransformation())
        m.Y = Normal.define_variable(mean=m.mu, variance=m.s, shape=(100,))

        return m

    def test_MAP(self):
        model_single = self.make_model_MAP()
        model_multi = self.make_model_MAP()
        mean_groundtruth = 3.
        variance_groundtruth = 5.
        data = np.random.randn(100) * np.sqrt(variance_groundtruth) + mean_groundtruth

        from mxfusion.inference import GradBasedInference, MAP

        infr_single = GradBasedInference(inference_algorithm=MAP(model=model_single, observed=[model_single.Y]))
        infr_single.run(Y=mx.nd.array(data, dtype='float64'), learning_rate=0.1, max_iter=2000, verbose=True,
                        multi_processor=False)
        mean_estimated_single = infr_single.params[model_single.mu].asnumpy()
        variance_estimated_single = infr_single.params[model_single.s].asnumpy()

        infr_multi = GradBasedInference(inference_algorithm=MAP(model=model_multi, observed=[model_multi.Y]))
        infr_multi.run(Y=mx.nd.array(data, dtype='float64'), learning_rate=0.1, max_iter=2000, verbose=True,
                       multi_processor=True)
        mean_estimated_double = infr_multi.params[model_multi.mu].asnumpy()
        variance_estimated_double = infr_multi.params[model_multi.s].asnumpy()


        rtol, atol = 1e-4, 1e-5

        assert np.allclose(mean_estimated_single, mean_estimated_double, rtol=rtol, atol=atol)
        assert np.allclose(variance_estimated_single, variance_estimated_double, rtol=rtol, atol=atol)


