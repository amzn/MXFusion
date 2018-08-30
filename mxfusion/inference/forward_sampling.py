from ..common.exceptions import InferenceError
from ..components.variables import Variable
from .variational import StochasticVariationalInference
from .inference_alg import InferenceAlgorithm
from .inference import TransferInference
from .map import MAP


class ForwardSamplingAlgorithm(InferenceAlgorithm):
    """
    The class of the forward sampling algorithm.

    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param model_graph: the definition of the probabilistic model
    :type model_graph: Model
    :param target_variables: (optional) the target variables to sample
    :type target_variables: [UUID]
    :param observed: A list of observed variables
    :type observed: [Variable]
    """

    def __init__(self, num_samples, model, observed, target_variables=None):
        super(ForwardSamplingAlgorithm, self).__init__(model=model,
                                                       observed=observed)
        self.num_samples = num_samples
        self._target_variables = target_variables

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
        samples = self.model.draw_samples(
            F=F, targets=self._target_variables, conditionals=knowns,
            num_samples=self.num_samples, constants=constants)
        if self._target_variables is not None:
            samples = {v: samples[v] for v in self._target_variables}
        samples = {k: v for k, v in samples.items()}
        return samples


class ForwardSampling(TransferInference):
    """
    Inference method of forward sampling.

    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param model_graph: the definition of the probabilistic model
    :type model_graph: Model
    :param observed: A list of observed variables
    :type observed: [Variable]
    :param var_ties: A dictionary of variables that are tied together and use the MXNet Parameter of the dict value's uuid.
    :type var_ties: { UUID to tie from : UUID to tie to }
    :param infr_params: list or single of InferenceParameters objects from previous Inference runs.
    :type infr_params: InferenceParameters or [InferenceParameters]
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
    :param inherited_inference: the inference method of which the model and inference results are taken
    :type inherited_inference: SVIInference or SVIMiniBatchInference
    :param observed: A list of observed variables
    :type observed: [Variable]
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
