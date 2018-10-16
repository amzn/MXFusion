from ..components import FunctionEvaluation, Distribution
from ..inference.inference_alg import SamplingAlgorithm
from ..modules.module import Module


class PredictionAlgorithm(SamplingAlgorithm):
    """
    The class of the forward sampling algorithm.

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
        for f in self.model.ordered_factors:
            if isinstance(f, FunctionEvaluation):
                outcome = f.eval(F=F, variables=variables,
                                 always_return_tuple=True)
                outcome_uuid = [v.uuid for _, v in f.outputs]
                for v, uuid in zip(outcome, outcome_uuid):
                    variables[uuid] = v
            elif isinstance(f, Distribution):
                known = [v in variables for _, v in f.outputs]
                if all(known):
                    continue
                elif any(known):
                    raise InferenceError("Part of the outputs of the distribution " + f.__class__.__name__ + " has been observed!")
                outcome_uuid = [v.uuid for _, v in f.outputs]
                outcome = f.draw_samples(
                    F=F, num_samples=self.num_samples, variables=variables, always_return_tuple=True)
                for v, uuid in zip(outcome, outcome_uuid):
                    variables[uuid] = v
            elif isinstance(f, Module):
                outcome = f.predict(
                    F=F, variables=variables)
                return outcome
