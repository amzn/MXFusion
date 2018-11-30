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
    def __init__(self, model, observed, cost_function, policy, n_time_steps, initial_state_generator, extra_graphs=None, num_samples=3, ctx=None, dtype=None):
        # TODO model should be a GP module
        super(ModelBasedAlgorithm, self).__init__(model, observed, extra_graphs=extra_graphs)
        self.cost_function = cost_function
        self.policy = policy
        self.initial_state_generator = initial_state_generator
        self.n_time_steps = n_time_steps
        self.num_samples = num_samples
        self.dtype = dtype if dtype is not None else get_default_dtype()
        self.mxnet_context = ctx if ctx is not None else get_default_device()


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
        s_0 = self.initial_state_generator(self.num_samples)
        a_0 = self.policy(s_0)
        a_t_plus_1 = a_0
        x_t = F.expand_dims(F.concat(s_0, a_0, dim=1), axis=1)
        cost = 0
        for t in range(self.n_time_steps):
            variables[self.model.X] = x_t
            res = self.model.Y.factor.predict(F, variables, targets=[self.model.Y], num_samples=self.num_samples)
            s_t_plus_1 = res[0]

            cost = cost + self.cost_function(s_t_plus_1, a_t_plus_1)

            a_t_plus_1 = mx.nd.expand_dims(self.policy(s_t_plus_1), axis=2)
            x_t = mx.nd.concat(s_t_plus_1, a_t_plus_1, dim=2)
        total_cost = F.sum(cost)
        return total_cost, total_cost


class PolicyUpdateGPParametricApprox(SamplingAlgorithm):
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
    def __init__(self, model, observed, cost_function, policy, n_time_steps, initial_state_generator, extra_graphs=None, num_samples=3, ctx=None, dtype=None, approx_samples=5000):
        # TODO model should be a GP module
        super(PolicyUpdateGPParametricApprox, self).__init__(model, observed, extra_graphs=extra_graphs)
        self.cost_function = cost_function
        self.policy = policy
        self.initial_state_generator = initial_state_generator
        self.n_time_steps = n_time_steps
        self.num_samples = num_samples
        self.dtype = dtype if dtype is not None else get_default_dtype()
        self.mxnet_context = ctx if ctx is not None else get_default_device()
        self.approx_samples = approx_samples

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
        s_0 = self.initial_state_generator(self.num_samples)
        a_0 = self.policy(s_0)
        x_t = F.expand_dims(F.concat(s_0, a_0, dim=1), axis=1)

        gp = self.model.Y.factor
        sample_func = gp.draw_parametric_samples(F, variables, self.num_samples, self.approx_samples)

        cost = 0
        for t in range(self.n_time_steps):
            s_t_plus_1 = sample_func(F, x_t)

            cost = cost + self.cost_function(s_t_plus_1)

            a_t_plus_1 = mx.nd.expand_dims(self.policy(s_t_plus_1), axis=2)
            x_t = mx.nd.concat(s_t_plus_1, a_t_plus_1, dim=2)
        total_cost = F.mean(cost)
        # assert total_cost.asscalar()<=0
        return total_cost, total_cost
