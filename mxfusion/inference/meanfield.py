from ..models.posterior import Posterior
from ..components.variables import PositiveTransformation
from ..components.variables import Variable, VariableType
from ..components.distributions.normal import Normal
from ..util.inference import variables_to_UUID


def create_Gaussian_meanfield(model, observed):
    """
    Create the Meanfield posterior for Variational Inference.

    :param model_graph: the definition of the probabilistic model
    :type model_graph: Model
    :param observed: A list of observed variables
    :type observed: [Variable]
    :returns: the resulting posterior representation
    :rtype: Posterior
    """
    observed = variables_to_UUID(observed)
    q = Posterior(model)
    for v in model.variables.values():
        if v.type == VariableType.RANDVAR and v not in observed:
            mean = Variable(shape=v.shape)
            variance = Variable(shape=v.shape,
                                transformation=PositiveTransformation())
            q[v].set_prior(Normal(mean=mean, variance=variance))
    return q
