# Copyright 2019 Amazon.com, Inc. or its affiliaeontext.default_ctx = mx.gpu(hvd.local_rank()) if mx.test_utils.list_gpus() else mx.cpu()
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
import mxnet.gluon.nn as nn
import pytest
hvd = pytest.importorskip("horovod.mxnet")
import mxnet as mx
import GPy

@pytest.mark.usefixtures("set_seed")
class TestDistributedBNN(object):
    """
        Test class that tests MXFusion Bayesian Neuran Network distributedly using Horovod with Stochastic Variational Inference (SVI).
        Run test with command "horovodrun -np {number_of_processors} -H localhost:4 pytest -v distributed_bnn_test.py".
        If run normally with pytest, the distributed training functionality won't be tested.
    """

    hvd.init()

    from mxfusion.common import config
    config.DEFAULT_DTYPE = 'float32'
    mx.context.Context.default_ctx = mx.gpu(hvd.local_rank()) if mx.test_utils.list_gpus() else mx.cpu()

    def make_neural_BNN(self, input, output, rand_num_scale):
        net = nn.HybridSequential(prefix='nn_')
        with net.name_scope():
            net.add(nn.Dense(input, activation="tanh", in_units=1))
            net.add(nn.Dense(input, activation="tanh", in_units=input))
            net.add(nn.Dense(output, flatten=True, in_units=input))
        net.initialize(mx.init.Xavier(magnitude=rand_num_scale))

        return net

    def make_model_BNN(self, net):
        from mxfusion import Variable, Model
        from mxfusion.components.functions import MXFusionGluonFunction
        from mxfusion.components.variables import PositiveTransformation
        from mxfusion.components.distributions import Normal

        m = Model()
        m.N = Variable()
        m.f = MXFusionGluonFunction(net, num_outputs=1, broadcastable=False)
        m.x = Variable(shape=(m.N, 1))
        m.v = Variable(shape=(1,), transformation=PositiveTransformation(), initial_value=mx.nd.array([0.01]))
        m.r = m.f(m.x)

        from mxfusion.components.functions.operators import broadcast_to

        for v in m.r.factor.parameters.values():
            v.set_prior(Normal(mean=broadcast_to(mx.nd.array([0], dtype=np.float32), v.shape),
                               variance=broadcast_to(mx.nd.array([1.], dtype=np.float32), v.shape)))
        return m

    def split_tuple_shape(self, tuple_var):
        list_tuple = list(tuple_var)
        list_tuple[0] = int(list_tuple[0]/hvd.size())
        t = tuple(list_tuple)

        return t

    def make_inference_BNN(self, model, net, x, y, num_samples=1000, distributed=False, minibatch=False, num_iter=2000, learning_rate=1e-1, batch_size=100):
        from mxfusion.inference import BatchInferenceLoop, MinibatchInferenceLoop, DistributedBatchInferenceLoop, DistributedMinibatchInferenceLoop, create_Gaussian_meanfield, GradBasedInference, StochasticVariationalInference, DistributedGradBasedInference
        dtype='float32'
        observed = [model.y, model.x]
        posterior = create_Gaussian_meanfield(model=model, observed=observed)

        alg = StochasticVariationalInference(num_samples=num_samples, model=model, posterior=posterior, observed=observed)

        if distributed:
            infr = DistributedGradBasedInference(inference_algorithm=alg, grad_loop=DistributedMinibatchInferenceLoop(batch_size=batch_size)) if minibatch else DistributedGradBasedInference(inference_algorithm=alg, grad_loop=DistributedBatchInferenceLoop())
        else:
            infr = GradBasedInference(inference_algorithm=alg, grad_loop=MinibatchInferenceLoop(batch_size=batch_size)) if minibatch else GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())

        shape_x = x.shape
        shape_y = y.shape

        if distributed:
            shape_x = self.split_tuple_shape(shape_x)
            shape_y = self.split_tuple_shape(shape_y)

        infr.initialize(y=shape_y,x=shape_x)
       
        for v_name, v in model.r.factor.parameters.items():
            infr.params[posterior[v].factor.mean] = net.collect_params()[v_name].data()
            infr.params[posterior[v].factor.variance] = mx.nd.ones_like(infr.params[posterior[v].factor.variance]) * 1e-6

        infr.run(max_iter=num_iter, learning_rate=learning_rate, y=mx.nd.array(y, dtype=dtype), x=mx.nd.array(x, dtype=dtype), verbose=True)

        return infr

    def prediction_resulting_BNN(self, model, infr, xt):

        from mxfusion.inference import VariationalPosteriorForwardSampling
        infr = VariationalPosteriorForwardSampling(10, [model.x], infr, [model.r])
        res = infr.run(x=mx.nd.array(xt, dtype='float32'))
        yt = res[0].asnumpy()

        return yt

    @pytest.mark.parametrize("N, num_samples, max_iter, learning_rate, bnn_input, bnn_output, bnn_rand_num_scale", [
        (200, 5, 10, 1e-1, 10, 2, 1),
        ])
    def test_BNN_classification(self, N, num_samples, max_iter, learning_rate, bnn_input, bnn_output, bnn_rand_num_scale):
        """
            Test the accuracy of distributing training of BNN classification with comparison to non-distributing training.
            This unit test specifically tests on Batch loop with SVI Inference for gradient optimisation.
            Parameters used for comparisons are mean and standard deviation of predicted BNN of the inferences.
        """

        k = GPy.kern.RBF(1, lengthscale=0.1)
        x = np.random.rand(N, 1)
        y = np.random.multivariate_normal(mean=np.zeros((N,)), cov=k.K(x), size=(1,)).T > 0.

        net = self.make_neural_BNN(input=bnn_input,output=bnn_output,rand_num_scale=bnn_rand_num_scale)

        from mxfusion.components.distributions import Categorical

        model_single = self.make_model_BNN(net)
        model_multi = self.make_model_BNN(net)

        model_single.y = Categorical.define_variable(log_prob=model_single.r, shape=(model_single.N, 1), num_classes=2)
        model_multi.y = Categorical.define_variable(log_prob=model_multi.r, shape=(model_multi.N, 1), num_classes=2)

        # Minibatch is set to False to run inference in batch loop
        infr_single = self.make_inference_BNN(model=model_single, net=net, num_samples=num_samples, x=x, y=y, distributed=False, minibatch=False, num_iter=max_iter, learning_rate=learning_rate)
        infr_multi = self.make_inference_BNN(model=model_multi, net=net, num_samples=num_samples, x=x, y=y,distributed=True, minibatch=False, num_iter=max_iter, learning_rate=learning_rate)

        xt = np.linspace(0, 1, 100)[:, None]

        prediction_single = self.prediction_resulting_BNN(model=model_single, infr=infr_single, xt=xt)
        prediction_multi = self.prediction_resulting_BNN(model=model_multi, infr=infr_multi, xt=xt)

        prediction_single_mean = prediction_single.mean(0)
        prediction_single_std = prediction_single.std(0)

        prediction_multi_mean = prediction_multi.mean(0)
        prediction_multi_std = prediction_multi.std(0)

        if max_iter < 200:
            rtol, atol = 10, 10
        else:
            rtol, atol = 1e-4, 1e-5

        assert np.allclose(prediction_single_mean, prediction_multi_mean, rtol=rtol, atol=atol)
        assert np.allclose(prediction_single_std, prediction_multi_std, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("N, num_samples, max_iter, learning_rate, bnn_input, bnn_output, bnn_rand_num_scale, batch_size", [
        (200, 5, 10, 1e-1, 10, 2, 1, 50)
        ])
    def test_BNN_classification_minibatch(self, N, num_samples, max_iter, learning_rate, bnn_input, bnn_output, bnn_rand_num_scale, batch_size):
        """
            Test the accuracy of distributing training of BNN classification with comparison to non-distributing training.
            This unit test specifically tests on MiniBatch loop with SVI Inference for gradient optimisation.
            Parameters used for comparisons are mean and standard deviation of predicted BNN of the inferences.
        """
        import GPy

        k = GPy.kern.RBF(1, lengthscale=0.1)
        x = np.random.rand(N, 1)
        y = np.random.multivariate_normal(mean=np.zeros((N,)), cov=k.K(x), size=(1,)).T > 0.

        net = self.make_neural_BNN(input=bnn_input,output=bnn_output,rand_num_scale=bnn_rand_num_scale)

        from mxfusion.components.distributions import Categorical

        model_single = self.make_model_BNN(net)
        model_multi = self.make_model_BNN(net)

        model_single.y = Categorical.define_variable(log_prob=model_single.r, shape=(model_single.N, 1), num_classes=2)
        model_multi.y = Categorical.define_variable(log_prob=model_multi.r, shape=(model_multi.N, 1), num_classes=2)

        # Minibatch is set to True to run inference in minibatch loop
        infr_single = self.make_inference_BNN(model=model_single, net=net, num_samples=num_samples, x=x, y=y, distributed=False, minibatch=True, num_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
        infr_multi = self.make_inference_BNN(model=model_multi, net=net, num_samples=num_samples, x=x, y=y,distributed=True, minibatch=True, num_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)

        xt = np.linspace(0, 1, 100)[:, None]

        prediction_single = self.prediction_resulting_BNN(model=model_single, infr=infr_single, xt=xt)
        prediction_multi = self.prediction_resulting_BNN(model=model_multi, infr=infr_multi, xt=xt)

        prediction_single_mean = prediction_single.mean(0)
        prediction_single_std = prediction_single.std(0)

        prediction_multi_mean = prediction_multi.mean(0)
        prediction_multi_std = prediction_multi.std(0)

        if max_iter < 200:
            rtol, atol = 10, 10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(prediction_single_mean, prediction_multi_mean, rtol=rtol, atol=atol)
        assert np.allclose(prediction_single_std, prediction_multi_std, rtol=rtol, atol=atol)
    
    @pytest.mark.parametrize("N, num_samples, max_iter, learning_rate, bnn_input, bnn_output, bnn_rand_num_scale, batch_size", [
        (200, 5, 10, 1e-1, 10, 2, 1, 50)
        ])
    def test_BNN_regression(self, N, num_samples, max_iter, learning_rate, bnn_input, bnn_output, bnn_rand_num_scale, batch_size):
        """
            Test the accuracy of distributing training of BNN regression with comparison to non-distributing training.
            This unit test specifically tests on Batch loop with SVI Inference for gradient optimisation.
            Parameters used for comparisons are mean and standard deviation of predicted BNN of the inferences.
        """

        k = GPy.kern.RBF(1, lengthscale=0.1)
        x = np.random.rand(N, 1)
        y = np.random.multivariate_normal(mean=np.zeros((N,)), cov=k.K(x), size=(1,)).T

        net = self.make_neural_BNN(bnn_input,bnn_output,bnn_rand_num_scale)

        model_single = self.make_model_BNN(net)
        model_multi = self.make_model_BNN(net)

        from mxfusion.components.distributions import Normal
        from mxfusion.components.functions.operators import broadcast_to

        model_single.y = Normal.define_variable(mean=model_single.r, variance=broadcast_to(model_single.v, (model_single.N, 1)), shape=(model_single.N, 1))
        model_multi.y = Normal.define_variable(mean=model_multi.r, variance=broadcast_to(model_multi.v, (model_multi.N, 1)), shape=(model_multi.N, 1))

        # Minibatch is set to False to run inference in batch loop.
        infr_single = self.make_inference_BNN(model=model_single, net=net, num_samples=num_samples, x=x, y=y,distributed=False, minibatch=False, num_iter=max_iter, learning_rate=learning_rate)
        infr_multi = self.make_inference_BNN(model=model_multi, net=net, num_samples=num_samples, x=x, y=y,distributed=True, minibatch=False, num_iter=max_iter, learning_rate=learning_rate)

        xt = np.linspace(0, 1, 100)[:, None]

        prediction_single = self.prediction_resulting_BNN(model=model_single, infr=infr_single, xt=xt)
        prediction_multi = self.prediction_resulting_BNN(model=model_multi, infr=infr_multi, xt=xt)

        prediction_single_mean = prediction_single.mean(0)
        prediction_single_std = prediction_single.std(0)

        prediction_multi_mean = prediction_multi.mean(0)
        prediction_multi_std = prediction_multi.std(0)

        if max_iter < 200:
            rtol, atol = 10, 10
        else:
            rtol, atol = 1e-4, 1e-5

        assert np.allclose(prediction_single_mean, prediction_multi_mean, rtol=rtol, atol=atol)
        assert np.allclose(prediction_single_std, prediction_multi_std, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("N, num_samples, max_iter, learning_rate, bnn_input, bnn_output, bnn_rand_num_scale, batch_size", [
        (1000, 3, 10, 1e-2, 50, 1, 3, 50)
        ])
    def test_BNN_regression_minibatch(self, N, num_samples, max_iter, learning_rate, bnn_input, bnn_output, bnn_rand_num_scale, batch_size):
        """
            Test the accuracy of distributing training of BNN regression with comparison to non-distributing training. Source and target must have the same data type when copying across device
            Parameters used for comparisons are mean and standard deviation of predicted BNN of the inferences.
        """

        k = GPy.kern.RBF(1, lengthscale=0.1)
        x = np.random.rand(N, 1)
        y = np.random.multivariate_normal(mean=np.zeros((N,)), cov=k.K(x), size=(1,)).T

        net = self.make_neural_BNN(bnn_input,bnn_output,bnn_rand_num_scale)

        model_single = self.make_model_BNN(net)
        model_multi = self.make_model_BNN(net)

        from mxfusion.components.distributions import Normal
        from mxfusion.components.functions.operators import broadcast_to

        model_single.y = Normal.define_variable(mean=model_single.r, variance=broadcast_to(model_single.v, (model_single.N, 1)), shape=(model_single.N, 1))
        model_multi.y = Normal.define_variable(mean=model_multi.r, variance=broadcast_to(model_multi.v, (model_multi.N, 1)), shape=(model_multi.N, 1))

        # Minibatch is set to True to run inference in minibatch loop
        infr_single = self.make_inference_BNN(model=model_single, net=net, num_samples=num_samples, x=x, y=y,distributed=False, minibatch=True, num_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
        infr_multi = self.make_inference_BNN(model=model_multi, net=net, num_samples=num_samples, x=x, y=y,distributed=True, minibatch=True, num_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)

        xt = np.linspace(0, 1, 100)[:, None]

        prediction_single = self.prediction_resulting_BNN(model=model_single, infr=infr_single, xt=xt)
        prediction_multi = self.prediction_resulting_BNN(model=model_multi, infr=infr_multi, xt=xt)

        prediction_single_mean = prediction_single.mean(0)
        prediction_single_std = prediction_single.std(0)

        prediction_multi_mean = prediction_multi.mean(0)
        prediction_multi_std = prediction_multi.std(0)

        if max_iter < 200:
            rtol, atol = 10, 10
        else:
            rtol, atol = 1e-4, 1e-5

        assert np.allclose(prediction_single_mean, prediction_multi_mean, rtol=rtol, atol=atol)
        assert np.allclose(prediction_single_std, prediction_multi_std, rtol=rtol, atol=atol)