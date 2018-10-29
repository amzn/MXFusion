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


from .inference_alg import InferenceAlgorithm
from ..components.variables import Variable, VariableType
from ..models.posterior import Posterior
from ..components.distributions import PointMass
from ..util.inference import variables_to_UUID


class MAP(InferenceAlgorithm):
    """
    The class of the Maximum A Posteriori (MAP) algorithm.

    :param model: the definition of the probabilistic model
    :type model: Model
    :param observed: A list of observed variables
    :type observed: [Variable]
    """
    def __init__(self, model, observed):
        posterior = MAP.create_posterior(model, variables_to_UUID(observed))
        super(MAP, self).__init__(model=model, observed=observed,
                                  extra_graphs=[posterior])

    @property
    def posterior(self):
        """
        return the variational posterior.
        """
        return self._extra_graphs[0]

    @staticmethod
    def create_posterior(model, observed):
        """
        Create the posterior of Maximum A Posteriori (MAP).

        :param model: the model definition
        :type model: Model
        :param observed: the list of observed variables
        :type observed: [UUID]
        :returns: the resulting posterior representation
        :rtype: Posterior
        """
        q = Posterior(model)
        for v in model.get_latent_variables(observed):
            q[v].assign_factor(PointMass(location=Variable(shape=v.shape)))
        return q

    def compute(self, F, variables):
        """
        The method for the computation of the inference algorithm

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param data: the data variables for inference
        :type data: {Variable: mxnet.ndarray.ndarray.NDArray or
            mxnet.symbol.symbol.Symbol}
        :param parameters: the parameters for inference
        :type parameters: {Variable: mxnet.ndarray.ndarray.NDArray or
            mxnet.symbol.symbol.Symbol}
        :param constants: the constants for inference
        :type parameters: {Variable: mxnet.ndarray.ndarray.NDArray or
            mxnet.symbol.symbol.Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.ndarray.ndarray.NDArray or mxnet.symbol.symbol.Symbol
        """
        for v in self.model.variables.values():
            if v.type == VariableType.RANDVAR and v not in self._observed:
                variables[v.uuid] = variables[self.posterior[v].factor.location.uuid]

        logL = self.model.log_pdf(F=F, variables=variables)
        return -logL, -logL
