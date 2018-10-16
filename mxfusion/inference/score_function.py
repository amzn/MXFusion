import mxnet as mx
from .inference_alg import InferenceAlgorithm
from .variational import StochasticVariationalInference
from ..components.variables import VariableType


class ScoreFunctionInference(StochasticVariationalInference):
    """
    Implemented following the [Black Box Variational Inference](https://arxiv.org/abs/1401.0118) paper.
    No control variates or Rao-Blackwellization included here so works for non-meanfield posteriors.

    Terminology:
      Lambda - Posterior parameters
      Theta - Model parameters

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
        super(ScoreFunctionInference, self).__init__(
            model=model, observed=observed, posterior=posterior, num_samples=num_samples)

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
        # Sample q(z | lambda)
        samples = self.posterior.draw_samples(
            F=F, variables=variables, num_samples=self.num_samples)
        variables.update(samples)

        q_z_lambda = self.posterior.log_pdf(F=F, variables=variables)

        p_x_z = self.model.log_pdf(F=F, variables=variables)

        difference_nograd = F.stop_gradient(p_x_z - q_z_lambda)
        gradient_lambda = F.mean(q_z_lambda * difference_nograd, axis=0)

        gradient_theta = F.mean(p_x_z - F.stop_gradient(q_z_lambda), axis=0)

        gradient_log_L = gradient_lambda + gradient_theta

        return -gradient_theta, -gradient_log_L


class ScoreFunctionRBInference(ScoreFunctionInference):
    """
    Implemented following the [Black Box Variational Inference](https://arxiv.org/abs/1401.0118) paper.

    The addition of Rao-Blackwellization and Control Variates (RBCV) requires that the posterior passed in be of meanfield form (i.e. all posterior variables independent.)

    Terminology:
      Lambda - Posterior parameters
      Theta - Model parameters

    Note: This is still in development and correctness is not fully tested.

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
        super(ScoreFunctionRBInference, self).__init__(
            model=model, observed=observed, posterior=posterior,
            num_samples=num_samples)

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
        # Sample q(z | lambda)
        samples = self.posterior.draw_samples(
            F=F, variables=variables, num_samples=self.num_samples)
        variables.update(samples)

        q_z_lambda = self.posterior.log_pdf(F=F, variables=variables)

        p_x_z = self.model.log_pdf(F=F, variables=variables)

        gradient_theta = F.mean(p_x_z - F.stop_gradient(q_z_lambda), axis=0)
        posterior_rvs = [v for v in self.posterior.variables.values() if v.type is VariableType.RANDVAR]
        f_list = []
        grad_list = []
        for i, v in enumerate(posterior_rvs):
            model_v = self.model[v]

            q_i_varset = self._extract_descendant_blanket_params(self.posterior, v)
            q_i_params = {key:val for key,val in variables.items() if key in q_i_varset}
            q_i = self.posterior.log_pdf(F=F, targets=q_i_params,
                                         variables=variables)

            p_i_varset = self._extract_descendant_blanket_params(self.model, model_v)
            p_i_params = {key:val for key,val in variables.items() if key in p_i_varset}
            p_i = self.model.log_pdf(F=F, targets=p_i_params,
                                     variables=variables)

            # TODO Remove this hack one day when MXNet doesn't have a bug?
            # Need to stop the gradient of p_i manually, for some reason it doesn't like
            # being used directly in this computation. Possibly only when p_i == p_x_z but
            # that is unconfirmed.
            f_i = q_i * (p_i.asscalar() - q_i.asscalar())

            # f_i = q_i * F.stop_gradient(p_i - q_i)
            f_list.append(F.expand_dims(f_i, axis=0))

            # With control variate?
            # h[i] = q_i
            # a = F.sum(f[i]) # covariance equation

            grad_i = F.mean(f_i, axis=0)
            grad_list.append(F.expand_dims(grad_i, axis=0))

        f = F.concat(*f_list, dim=0)
        grad = F.concat(*grad_list, dim=0)

        gradient_lambda = F.sum(grad)


        # Robbins-Monro sequence??
        gradient_log_L = gradient_lambda + gradient_theta

        return -gradient_theta, -gradient_log_L

    def _extract_descendant_blanket_params(self, graph, node):
        """
        Returns a set of the markov blankets of all of the descendents of the node in the graph, mapped to their parameter form.
        """
        if node.graph != graph.components_graph:
            raise InferenceError("Graph of node and graph to find it's descendents in differ. These should match so something went wrong.")

        descendents = graph.get_descendants(node)
        varset = [graph.get_markov_blanket(d) for d in descendents]
        varset = set(item for s in varset for item in s)
        return varset
