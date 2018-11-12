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


from ..common.exceptions import InferenceError
from ..components.variables import Variable, VariableType
from .variational import StochasticVariationalInference
from .inference_alg import SamplingAlgorithm
from .inference import TransferInference
from .map import MAP
from ..components.variables.runtime_variable import expectation


class ExpectationAlgorithm(SamplingAlgorithm):
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
        samples = self.model.draw_samples(
            F=F, variables=variables,
            num_samples=self.num_samples)
        samples = {k: expectation(F,v) for k, v in samples.items()}

        if self.target_variables:
            return tuple(samples[v] for v in self.target_variables)
        else:
            return samples


class ExpectationScoreFunctionAlgorithm(SamplingAlgorithm):
    """
    Sampling-based inference algorithm that computes the expectation of the model w.r.t. some loss function in that model, specified as the target variable. It does so via the score function trick sampling the necessary inputs to the function and using them to compute a Monte Carlo estimate of the loss function's gradient.

    :param model: the definition of the probabilistic model
    :type model: Model
    :param observed: A list of observed variables
    :type observed: [Variable]
    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param target_variables: the target function in the model to optimize. should only be one for this.
    :type target_variables: [UUID]
    :param extra_graphs: a list of extra FactorGraph used in the inference
                         algorithm.
    :type extra_graphs: [FactorGraph]
    """
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
        samples = self.model.draw_samples(
            F=F, variables=variables,
            num_samples=self.num_samples)
        variables.update(samples)
        targets = [v for v in self.model.get_latent_variables(self.observed_variables) if v.type == VariableType.RANDVAR]

        q_z_lambda = self.model.log_pdf(F=F, variables=variables, targets=targets)

        p_x_z = variables[self.target_variables[0]]

        gradient_lambda = F.mean(q_z_lambda * F.stop_gradient(p_x_z), axis=0)

        gradient_theta = F.mean(p_x_z, axis=0) # TODO known issue. This will double count the gradient of any distribution using the reparameterization trick (i.e. Normal). Issue #91

        gradient_log_L = gradient_lambda + gradient_theta

        return gradient_theta, gradient_log_L
