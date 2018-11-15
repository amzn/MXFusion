
from ..common.exceptions import InferenceError
from ..common.config import DEFAULT_DTYPE
from ..components.variables import Variable
from .variational import StochasticVariationalInference
from .inference_alg import SamplingAlgorithm
from .inference import Inference
from .map import MAP
from ..components.distributions import Normal
from ..models import Posterior
import mxnet as mx
from ..components.distributions.random_gen import MXNetRandomGenerator

class MetropolisHastingsAlgorithm(SamplingAlgorithm):
    """
    The Metropolis-Hastings MCMCsampling algorithm.

    :param model: the definition of the probabilistic model
    :type model: Model
    :param proposal: the proposal distribution to draw comparison samples against.
    :type proposal: Distribution
    :param observed: A list of observed variables
    :type observed: [Variable]
    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param target_variables: (optional) the target variables to sample
    :type target_variables: [UUID]
    """
    def __init__(self, model, observed, proposal=None, num_samples=1,
                 target_variables=None, variance=None, rand_gen=None, dtype=None):
        self._rand_gen = MXNetRandomGenerator if rand_gen is None else \
            rand_gen
        self._dtype = dtype if dtype is not None else DEFAULT_DTYPE
        self._copy_map = {}
        self._proposals_chosen = 0
        self.variance = variance if variance is not None else mx.nd.array([1.], dtype=self._dtype)
        if proposal is None:
            proposal = Posterior(model)
            for rv in model.get_latent_variables(observed):
                rv_previous = Variable(shape=rv.shape)
                self._copy_map[rv.uuid] = rv_previous.uuid
                proposal[rv].set_prior(Normal(mean=rv_previous, variance=self.variance))

        super(MetropolisHastingsAlgorithm, self).__init__(
            model=model, observed=observed, num_samples=num_samples,
            target_variables=target_variables, extra_graphs=[proposal])

    def compute(self, F, variables):
        """
        Returns Metropolis-Hastings samples.
        :param x: {'uuid': latest sample}
        :rtype: {'uuid': next sample}
        """
        x = variables.copy()
        x = {k: v for k,v in x.items() if k not in self._copy_map}
        # draw proposal samples using the last steps output
        x_proposal = self.proposal.draw_samples(F, variables=x, num_samples=1)
        x_proposal_full = x_proposal.copy()
        # compute new ratio
        x_proposal_full.update({k:v for k,v in x.items() if k not in x_proposal})
        # proposal_new = self.proposal.log_pdf(F, x_proposal_full)
        # swapped = swap the uuids of new and old latent variables
        # proposal_old = self.proposal.log_pdf(F, swapped)
        # import pdb; pdb.set_trace()
        model_new = self.model.log_pdf(F, x_proposal_full)
        model_old = self.model.log_pdf(F, variables)
        alpha = (model_new - model_old)
        # alpha += (proposal_old - proposal_new)
        r_min = F.exp(F.minimum(mx.nd.array([0], dtype=self._dtype),alpha))

        unif_sample = self._rand_gen.sample_uniform(0,1)

        # return this step's samples based on ratio
        if unif_sample < r_min:
            is_proposal = True
            return_choice = x_proposal_full
        else:
            return_choice = variables
            is_proposal = False
        return_subset = {k:v for k,v in return_choice.items() if k in x_proposal}
        # print("Original : {} \n Proposal : {} \n : z_copy {}\n".format(variables[self.proposal.z.uuid].asnumpy(), x_proposal[self.proposal.z.uuid].asnumpy(), variables[self.proposal.z.factor.mean.uuid].asnumpy()))
        # import pdb; pdb.set_trace()
        # print("unif {} : r_min {} : alpha {}".format(unif_sample.asscalar(), r_min.asscalar(), alpha.asscalar()))
        # print("q(old) {:.3f} - q(new) {:.3f} + m(old) {:.3f} - m(new) : {:.3f}  \n".format(proposal_old.asscalar(), proposal_new.asscalar(), model_old.asscalar(), model_new.asscalar()))
        # import pdb; pdb.set_trace()
        return return_subset, is_proposal

    @property
    def proposal(self):
        """
        Return the proposal distribution
        """
        return self._extra_graphs[0]


class MCMCInference(Inference):
    """
    The abstract class for MCMC-based inference methods.
    An inference method consists of a few components: the applied inference algorithm, the model definition, and the inference parameters.

    :param inference_algorithm: The applied inference algorithm
    :type inference_algorithm: InferenceAlgorithm
    :param graphs: a list of graph definitions required by the inference method. It includes the model definition and necessary posterior approximation.
    :type graphs: [FactorGraph]
    :param observed: A list of observed variables
    :type observed: [Variable]
    :param constants: Specify a list of model variables as constants
    :type constants: {Variable: mxnet.ndarray}
    :param hybridize: Whether to hybridize the MXNet Gluon block of the inference method.
    :type hybridize: boolean
    :param dtype: data type for internal numerical representation
    :type dtype: {numpy.float64, numpy.float32, 'float64', 'float32'}
    :param context: The MXNet context
    :type context: {mxnet.cpu or mxnet.gpu}
    """
    def __init__(self, inference_algorithm, constants=None,
                 hybridize=False, dtype=None, context=None):
        super(MCMCInference, self).__init__(
            inference_algorithm=inference_algorithm, constants=constants,
            hybridize=hybridize, dtype=dtype, context=context)
        self._number_proposals = 0
        self.samples = {}

    def create_executor(self):
        """
        Return a MXNet Gluon block responsible for the execution of the inference method.
        """
        infr = self._inference_algorithm.create_executor(
            data_def=self.observed_variable_UUIDs, params=self.params,
            var_ties=self.params.var_ties, rv_scaling=None)
        if self._hybridize:
            infr.hybridize()
        infr.initialize(ctx=self.mxnet_context)
        return infr

    def run(self, optimizer='adam', learning_rate=1e-3, max_iter=2000,
            verbose=False, n_prints=10, **kwargs):
        """
        Run the inference method.

        :param optimizer: the choice of optimizer (default: 'adam')
        :type optimizer: str
        :param learning_rate: the learning rate of the gradient optimizer (default: 0.001)
        :type learning_rate: float
        :param max_iter: the maximum number of iterations of gradient optimization
        :type max_iter: int
        :param verbose: whether to print per-iteration messages.
        :type verbose: boolean
        :param **kwargs: The keyword arguments specify the data for inferences. The key of each argument is the name of the corresponding
            variable in model definition and the value of the argument is the data in numpy array format.
        """
        data = [kwargs[v] for v in self.observed_variable_names]
        self.initialize(**kwargs)

        self.params._params.initialize(ctx=self.mxnet_context)
        infr = self.create_executor()
        iter_step = max(max_iter // n_prints, 1)
        self._number_proposals = 0
        number_proposals = 0
        for i in range(max_iter):
            sample, is_proposal = infr(mx.nd.zeros(1), *data)
            if is_proposal:
                number_proposals += 1
            for k,v in sample.items():
                if k not in self.samples:
                    self.samples[k] = v
                else:
                    self.samples[k] = mx.nd.concat(self.samples[k], v, dim=0)
                v_shaped = mx.nd.reshape(v, shape=self.params._params.get(k).data().shape)
                self.params._params.get(k).set_data(v_shaped)
                self.params._params.get(self.inference_algorithm._copy_map[k]).set_data(v_shaped)
            if i > 0 and i % iter_step == 0:
                self._number_proposals += number_proposals
                if verbose:
                    print("{}th step. Acceptance rate so far: {:.3f}, latest {}".format(i, self._number_proposals / i, number_proposals / iter_step))
                number_proposals = 0

        return self.samples
