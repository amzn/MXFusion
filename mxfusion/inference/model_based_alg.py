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
    def __init__(self, model, observed, cost_function, policy, n_time_steps, s_0, extra_graphs=None):
        # model is a GP module
        super(ModelBasedAlgorithm, self).__init__(model, observed, extra_graphs=extra_graphs)
        self.cost_function = cost_function
        self.policy = policy
        self.s_0 = s_0
        self.n_time_steps = n_time_steps

    def compute(self, F, variables):
        """
        Compute the inference algorithm

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of
        variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.ndarray.ndarray.NDArray or mxnet.symbol.symbol.Symbol
        """

        from mxfusion.inference import TransferInference, ModulePredictionAlgorithm
        infr_pred = TransferInference(ModulePredictionAlgorithm(model=self.model, observed=[self.model.X],
                                                                target_variables=[self.model.Y]),
                                      infr_params=self.infr.params)

        a_0 = self.policy(self.s_0)
        x_t = mx.nd.concatenate([self.s_0, a_0], axis=1)
        cost = mx.nd.array([0.])
        for t in range(self.n_time_steps):

            # s_t_plus_1, _ = self.model.GP.predict(x_t)

            variables[model.X] = x_t
            res = self.model.predict(F, variables, targets=[self.model.Y])
            # res = infr_pred.run(X=x_t)
            # s_t_plus_1_var = res[1].asnumpy()[0]
            s_t_plus_1 = res[0].asnumpy()[0]

            cost = cost.concat(self.cost_function(s_t_plus_1))

            a_t_plus_1 = self.policy(s_t_plus_1)
            x_t = mx.nd.concatenate([s_t_plus_1, a_t_plus_1], axis=1)
        return F.sum(cost)
