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

from .inference_alg import SamplingAlgorithm
from ..common.config import get_default_dtype, get_default_device


class PILCOAlgorithm(SamplingAlgorithm):
    """
    Sampling-based inference algorithm that returns the expectation of each variable in the model.

    :param model: the definition of the probabilistic model
    :type model: Model
    :param observed: A list of observed variables
    :type observed: [Variable]
    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param target_variables: (optional) the target variables to sample
    :type target_variables: [UUID]
    :param extra_graphs: a list of extra FactorGraph used in the inference
                         algorithm.
    :type extra_graphs: [FactorGraph]
    """
    def __init__(self, model, observed, cost_function, policy, n_time_steps, initial_state_generator, extra_graphs=None, num_samples=3, ctx=None, dtype=None):
        """
        :param model: The model to use to generate the next state from a state/action pair.
        :param observed: Observed variables for the model.
        :param cost_function: The cost function to evaluate state/action pairs on.
        :param policy: The policy function to determine what action to take next from a particular state.
        :param n_time_steps: How many time steps to roll forward using the model+policy to generate a trajectory.
        :param initial_state_generator: Function that generates initial states for the model to begin at.
        :param num_samples: How many sample trajectories to compute at once
        """
        super(PILCOAlgorithm, self).__init__(model, observed, extra_graphs=extra_graphs)
        self.cost_function = cost_function
        self.policy = policy
        self.initial_state_generator = initial_state_generator
        self.n_time_steps = n_time_steps
        self.num_samples = num_samples
        self.dtype = dtype if dtype is not None else get_default_dtype()
        self.mxnet_context = ctx if ctx is not None else get_default_device()


    def compute(self, F, variables):
        """
        Compute the PILCO algorithm's policy computation loop.

        1. Generates a number of initial state + action pairs
        2. For each state+action pair:
          1. Predict the new state (s_t_plus_1) given the current state and action pair
          2. Compute the cost of being in that state
          3. Use the policy to compute the next action (a_t_plus_1) to take from s_t_plus_1
          4. Repeat n_time_steps into the future, using the previous round's state/action pairs to roll forward.
        3. Return the total cost of all sample trajectories over time.

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of
        variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.NDArray or mxnet.Symbol
        """
        s_0 = self.initial_state_generator(self.num_samples)
        a_0 = self.policy(s_0)
        a_t_plus_1 = a_0
        x_t = F.expand_dims(F.concat(s_0, a_0, dim=1), axis=1)
        cost = 0
        for t in range(self.n_time_steps):
            variables[self.model.X] = x_t
            res = self.model.Y.factor.predict(F, variables, targets=[self.model.Y], num_samples=self.num_samples)[0]
            s_t_plus_1 = res[0]

            cost = cost + self.cost_function(s_t_plus_1, a_t_plus_1)

            a_t_plus_1 = mx.nd.expand_dims(self.policy(s_t_plus_1), axis=2)
            x_t = mx.nd.concat(s_t_plus_1, a_t_plus_1, dim=2)
        total_cost = F.sum(cost)
        return total_cost, total_cost
