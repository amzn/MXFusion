from ..common.exceptions import InferenceError
from ..components.variables import Variable
from .variational import StochasticVariationalInference
from .inference_alg import SamplingAlgorithm
from .inference import TransferInference
from .map import MAP


class ForwardSamplingAlgorithm(SamplingAlgorithm):
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
        samples = self.model.draw_samples(
            F=F, variables=variables, targets=self.target_variables,
            num_samples=self.num_samples)

        return samples


class ForwardSampling(TransferInference):
    """
    Inference method of forward sampling.

    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param model: the definition of the probabilistic model
    :type model: Model
    :param observed: A list of observed variables
    :type observed: [Variable]
    :param var_ties: A dictionary of variables that are tied together and use the MXNet Parameter of the dict value's uuid.
    :type var_ties: { UUID to tie from : UUID to tie to }
    :param infr_params: list or single of InferenceParameters objects from previous Inference runs.
    :type infr_params: InferenceParameters or [InferenceParameters]
    :param target_variables: (optional) the target variables to sample
    :type target_variables: [Variable]
    :param hybridize: Whether to hybridize the MXNet Gluon block of the inference method.
    :type hybridize: boolean
    :param constants: Specify a list of model variables as constants
    :type constants: {Variable: mxnet.ndarray}
    :param dtype: data type for internal numberical representation
    :type dtype: {numpy.float64, numpy.float32, 'float64', 'float32'}
    :param context: The MXNet context
    :type context: {mxnet.cpu or mxnet.gpu}
    """
    def __init__(self, num_samples, model, observed, var_tie, infr_params,
                 target_variables=None, hybridize=False, constants=None,
                 dtype=None, context=None):
        if target_variables is not None:
            target_variables = [v.uuid for v in target_variables if
                                isinstance(v, Variable)]

        infr = ForwardSamplingAlgorithm(
            num_samples=num_samples, model=model, observed=observed,
            target_variables=target_variables)
        super(ForwardSampling, self).__init__(
            inference_algorithm=infr, var_tie=var_tie, infr_params=infr_params,
            constants=constants, hybridize=hybridize, dtype=dtype,
            context=context)


def merge_posterior_into_model(model, posterior, observed):
    """
    Replace the prior distributions of a model with its variational posterior distributions.

    :param model: the definition of the probabilistic model
    :type model: Model
    :param posterior: the definition of the variational posterior of the probabilistic model
    :param posterior: Posterior
    :param observed: A list of observed variables
    :type observed: [Variable]
    """
    new_model, var_map = model.clone()
    for lv in model.get_latent_variables(observed):
        v = posterior.extract_distribution_of(posterior[lv])
        new_model.replace_subgraph(new_model[v], v)
    return new_model, var_map


class VariationalPosteriorForwardSampling(ForwardSampling):
    """
    The forward sampling method for variational inference.

    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param observed: A list of observed variables
    :type observed: [Variable]
    :param inherited_inference: the inference method of which the model and inference results are taken
    :type inherited_inference: SVIInference or SVIMiniBatchInference
    :param target_variables: (optional) the target variables to sample
    :type target_variables: [Variable]
    :param constants: Specify a list of model variables as constants
    :type constants: {Variable: mxnet.ndarray}
    :param hybridize: Whether to hybridize the MXNet Gluon block of the inference method.
    :type hybridize: boolean
    :param dtype: data type for internal numberical representation
    :type dtype: {numpy.float64, numpy.float32, 'float64', 'float32'}
    :param context: The MXNet context
    :type context: {mxnet.cpu or mxnet.gpu}
    """
    def __init__(self, num_samples, observed,
                 inherited_inference, target_variables=None,
                 hybridize=False, constants=None, dtype=None, context=None):
        if not isinstance(inherited_inference.inference_algorithm,
                          (StochasticVariationalInference, MAP)):
            raise InferenceError('inherited_inference needs to be a subclass of SVIInference or SVIMiniBatchInference.')

        m = inherited_inference.inference_algorithm.model
        q = inherited_inference.inference_algorithm.posterior

        model_graph, var_map = merge_posterior_into_model(
            m, q, observed=inherited_inference.observed_variables)

        super(VariationalPosteriorForwardSampling, self).__init__(
            num_samples=num_samples, model=model_graph,
            observed=observed,
            var_tie={}, infr_params=inherited_inference.params,
            target_variables=target_variables, hybridize=hybridize,
            constants=constants, dtype=dtype, context=context)
