from .inference_alg import InferenceAlgorithm


class StochasticVariationalInference(InferenceAlgorithm):
    """
    The class of the Stochastic Variational Inference (SVI) algorithm.

    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param model: the definition of the probabilistic model
    :type model: Model
    :param posterior: the definition of the variational posterior of the probabilistic model
    :param posterior: Posterior
    :param observed: A list of observed variables
    :type observed: [Variable]
    """

    def __init__(self, num_samples, model, posterior, observed):
        super(StochasticVariationalInference, self).__init__(
            model=model, observed=observed, extra_graphs=[posterior])
        self.num_samples = num_samples

    @property
    def posterior(self):
        """
        return the variational posterior.
        """
        return self._extra_graphs[0]

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
        knowns.update(parameters)
        knowns.update(constants)

        samples = self.posterior.draw_samples(
            F=F, conditionals=knowns, num_samples=self.num_samples,
            constants=constants)
        knowns.update(samples)
        logL = self.model.compute_log_prob(
            F=F, targets=knowns, constants=constants)
        logL = logL - self.posterior.compute_log_prob(
            F=F, targets=knowns, constants=constants)
        return -logL
