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


from ..components import FunctionEvaluation, Distribution
from ..inference.inference_alg import SamplingAlgorithm
from ..modules.module import Module
from ..common.exceptions import InferenceError


class ModulePredictionAlgorithm(SamplingAlgorithm):
    """
    A prediction algorithm for modules. The algorithm evaluates all the functions, draws samples from distributions and runs the predict method on all the modules.

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
        outcomes = {}
        for f in self.model.ordered_factors:
            if isinstance(f, FunctionEvaluation):
                outcome = f.eval(F=F, variables=variables,
                                 always_return_tuple=True)
                outcome_uuid = [v.uuid for _, v in f.outputs]
                for v, uuid in zip(outcome, outcome_uuid):
                    variables[uuid] = v
                    outcomes[uuid] = v
            elif isinstance(f, Distribution):
                known = [v in variables for _, v in f.outputs]
                if all(known):
                    continue
                elif any(known):
                    raise InferenceError("Part of the outputs of the distribution " + f.__class__.__name__ + " has been observed!")
                outcome_uuid = [v.uuid for _, v in f.outputs]
                outcome = f.draw_samples(
                    F=F, num_samples=self.num_samples, variables=variables,
                    always_return_tuple=True)
                for v, uuid in zip(outcome, outcome_uuid):
                    variables[uuid] = v
                    outcomes[uuid] = v
            elif isinstance(f, Module):
                outcome_uuid = [v.uuid for _, v in f.outputs]
                outcome = f.predict(
                    F=F, variables=variables, targets=outcome_uuid,
                    num_samples=self.num_samples)
                for v, uuid in zip(outcome, outcome_uuid):
                    variables[uuid] = v
                    outcomes[uuid] = v
        if self.target_variables:
            return tuple(outcomes[uuid] for uuid in self.target_variables)
        else:
            return outcomes
