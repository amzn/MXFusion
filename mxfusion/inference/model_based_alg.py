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


class ModelBasedAlgorithm(SamplingAlgorithm):
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
    def __init__(self, model, observed, cost_function, policy, n_time_steps, initial_state, extra_graphs=None, num_samples=3):
        # TODO model should be a GP module
        super(ModelBasedAlgorithm, self).__init__(model, observed, extra_graphs=extra_graphs)
        self.cost_function = cost_function
        self.policy = policy
        self.s_0 = initial_state
        self.n_time_steps = n_time_steps
        self.num_samples = num_samples

    def compute(self, F, variables):
        """
        Compute the inference algorithm

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of
        variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.NDArray or mxnet.Symbol
        """
        a_0 = self.policy(self.s_0)
        x_t_pre = mx.nd.expand_dims(mx.nd.concat(self.s_0, a_0, dim=1), axis=0)
        x_t = mx.nd.broadcast_to(x_t_pre, shape=(self.num_samples,) + x_t_pre.shape[1:])
        cost = mx.nd.zeros(shape=(self.num_samples,1), dtype='float64')
        for t in range(self.n_time_steps):

            variables[self.model.X] = x_t
            res = self.model.Y.factor.predict(F, variables, targets=[self.model.Y])
            s_t_mean = res[0][0]
            s_t_std = mx.nd.sqrt(res[0][1])
            s_t_std = mx.nd.expand_dims(mx.nd.broadcast_axis(s_t_std, axis=1, size=self.s_0.shape[-1]), axis=1)
            die = F.random.normal(shape=s_t_mean.shape, dtype='float64')
            s_t_plus_1 = s_t_mean + s_t_std * die
            # s_t_plus_1 = F.random.normal(loc=s_t_mean, scale=s_t_std, dtype='float64')
            # import pdb; pdb.set_trace()
            # print(s_t_plus_1)

            cost = mx.nd.concat(cost, self.cost_function(s_t_plus_1), dim=1)

            a_t_plus_1 = mx.nd.expand_dims(self.policy(s_t_plus_1), axis=2)
            x_t = mx.nd.concat(s_t_plus_1, a_t_plus_1, dim=2)
            # import pdb; pdb.set_trace()
        total_cost = F.sum(cost)
        return total_cost, total_cost
