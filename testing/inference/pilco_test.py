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

import numpy as np
import pytest
import mxnet as mx
from mxfusion import Model, Variable
from mxfusion.components.variables import PositiveTransformation
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.modules.gp_modules import GPRegression
from mxfusion.inference import GradBasedInference, MAP
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import Dense
from mxfusion.inference import GradTransferInference
from mxfusion.inference.pilco_alg import PolicyUpdateGPParametricApprox

class NNController(HybridBlock):
    def __init__(self, prefix=None, params=None, in_units=100, obs_space_high=3):
        super(NNController, self).__init__(prefix=prefix, params=params)
        self.dense1 = Dense(100, in_units=obs_space_high, activation='relu')
        self.dense2 = Dense(1, in_units=100, activation='tanh')
    def hybrid_forward(self, F, x):
        out = self.dense2(self.dense1(x))*2
        return out

class CostFunction(mx.gluon.HybridBlock):
    """
    The goal is to get the pendulum upright and stable as quickly as possible.
    Taken from the code for Pendulum.
    """
    def hybrid_forward(self, F, state, action):
        """
        :param state: [np.cos(theta), np.sin(theta), ~ momentum(theta)]
        a -> 0 when pendulum is upright, largest when pendulum is hanging down completely.
        b -> penalty for taking action
        c -> penalty for pendulum momentum
        """
        a_scale = 2.
        b_scale = .001
        c_scale = .1
        a = F.sum(a_scale * (state[:,:,0:1] -1) ** 2, axis=-1)
        b = F.sum(b_scale * action ** 2, axis=-1)
        c = F.sum(c_scale * state[:,:,2:3] ** 2, axis=-1)
        return (a + c + b)

@pytest.mark.usefixtures("set_seed")
class TestPILCOInference(object):
    """
    Test class that tests the MXFusion.inference.PILCOAlgorithm and MXFusion.inferenceGradTransferInference classes.
    """

    def run_one_episode(self):
        reward_shape = (1,)
        observations_shape = (200,3) # steps, obs shape
        actions_shape = (199,1) # steps - 1, action shape
        return np.random.rand(*reward_shape), np.random.rand(*observations_shape), np.random.rand(*actions_shape)

    def prepare_data(self, state_list, action_list, win_in):
        """
        Prepares a list of states and a list of actions as inputs to the Gaussian Process for training.
        """

        X_list = []
        Y_list = []

        for state_array, action_array in zip(state_list, action_list):
            # the state and action array shape should be aligned.
            assert state_array.shape[0]-1 == action_array.shape[0]

            for i in range(state_array.shape[0]-win_in):
                Y_list.append(state_array[i+win_in:i+win_in+1])
                X_list.append(np.hstack([state_array[i:i+win_in].flatten(), action_array[i:i+win_in].flatten()]))
        X = np.vstack(X_list)
        Y = np.vstack(Y_list)
        return X, Y


    def fit_model(self, state_list, action_list, win_in, verbose=True, max_iter=1000):
        """
        Fits a Gaussian Process model to the state / action pairs passed in.
        This creates a model of the environment which is used during
        policy optimization instead of querying the environment directly.

        See mxfusion.gp_modules for additional types of GP models to fit,
        including Sparse GP and Stochastic Varitional Inference Sparse GP.
        """
        X, Y = self.prepare_data(state_list, action_list, win_in)

        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, X.shape[-1]))
        m.noise_var = Variable(shape=(1,), transformation=PositiveTransformation(),
                               initial_value=0.01)
        m.kernel = RBF(input_dim=X.shape[-1], variance=1, lengthscale=1, ARD=True)
        m.Y = GPRegression.define_variable(
            X=m.X, kernel=m.kernel, noise_var=m.noise_var,
            shape=(m.N, Y.shape[-1]))
        m.Y.factor.gp_log_pdf.jitter = 1e-6

        infr = GradBasedInference(
            inference_algorithm=MAP(model=m, observed=[m.X, m.Y]))
        infr.run(X=mx.nd.array(X),
                 Y=mx.nd.array(Y),
                 max_iter=max_iter, learning_rate=0.1, verbose=verbose)
        return m, infr, X, Y

    def optimize_policy(self, policy, cost_func, model, infr,
                        model_data_X, model_data_Y,
                        initial_state_generator, num_grad_steps,
                        learning_rate=1e-2, num_time_steps=100,
                        num_samples=10, verbose=True):
        """
        Takes as primary inputs a policy, cost function, and trained model.
        Optimizes the policy for num_grad_steps number of iterations.
        """
        mb_alg = PolicyUpdateGPParametricApprox(
            model=model, observed=[model.X, model.Y],
            cost_function=cost_func,
            policy=policy, n_time_steps=num_time_steps,
            initial_state_generator=initial_state_generator,
            num_samples=num_samples)

        infr_pred = GradTransferInference(
            mb_alg, infr_params=infr.params, train_params=policy.collect_params())
        infr_pred.run(
            max_iter=num_grad_steps,
            X=mx.nd.array(model_data_X),
            Y=mx.nd.array(model_data_Y),
            verbose=verbose, learning_rate=learning_rate)
        return policy

    def initial_state_generator(self, num_initial_states, obs_space_shape=3):
        """
        Starts from valid states by drawing theta and momentum
        then computing np.cos(theta) and np.sin(theta) for state[0:2].s
        """
        return mx.nd.array(
            [np.random.rand(obs_space_shape) for i in range(num_initial_states)])

    def test_pilco_basic_passthrough(self):
        policy = NNController()
        policy.collect_params().initialize(mx.initializer.Xavier(magnitude=1))
        cost = CostFunction()
        num_episode = 2 # how many model fit + policy optimization episodes to run
        num_samples = 2 # how many sample trajectories the policy optimization loop uses
        num_grad_steps = 2 # how many gradient steps the optimizer takes per episode
        num_time_steps = 2 # how far to roll out each sample trajectory
        learning_rate = 1e-3 # learning rate for the policy optimization

        all_states = []
        all_actions = []

        for i_ep in range(num_episode):
            # Run an episode and collect data.
            policy_func = lambda x: policy(mx.nd.expand_dims(mx.nd.array(x), axis=0)).asnumpy()[0]
            total_reward, states, actions = self.run_one_episode()
            all_states.append(states)
            all_actions.append(actions)

            # Fit a model.
            model, infr, model_data_X, model_data_Y = self.fit_model(
                all_states, all_actions, win_in=1, verbose=True, max_iter=5)

            # Optimize the policy.
            policy = self.optimize_policy(
                policy, cost, model, infr, model_data_X, model_data_Y,
                self.initial_state_generator, num_grad_steps=num_grad_steps,
                num_samples=num_samples, learning_rate=learning_rate,
                num_time_steps=num_time_steps)
