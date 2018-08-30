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

    def compute(self, F, data, parameters, constants):
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
        knowns = data.copy()
        knowns.update(constants)

        for v in self.model.variables.values():
            if v.type == VariableType.RANDVAR and v not in data:
                # TODO: check self._post_graph[v].factor is a PointMass distribution
                knowns[v.uuid] = parameters[self.posterior[v].factor.location.uuid]
            elif v.type == VariableType.PARAMETER and v not in data and v not in constants:
                if not v.isInherited:
                    knowns[v.uuid] = parameters[v.uuid]

        logL = self.model.compute_log_prob(F=F, targets=knowns,
                                           constants=constants)
        return -logL
