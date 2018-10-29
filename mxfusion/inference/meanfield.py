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


from ..models.posterior import Posterior
from ..components.variables import PositiveTransformation
from ..components.variables import Variable, VariableType
from ..components.distributions.normal import Normal
from ..util.inference import variables_to_UUID
from ..common.config import get_default_dtype


def create_Gaussian_meanfield(model, observed, dtype=None):
    """
    Create the Meanfield posterior for Variational Inference.

    :param model_graph: the definition of the probabilistic model
    :type model_graph: Model
    :param observed: A list of observed variables
    :type observed: [Variable]
    :returns: the resulting posterior representation
    :rtype: Posterior
    """
    dtype = get_default_dtype() if dtype is None else dtype
    observed = variables_to_UUID(observed)
    q = Posterior(model)
    for v in model.variables.values():
        if v.type == VariableType.RANDVAR and v not in observed:
            mean = Variable(shape=v.shape)
            variance = Variable(shape=v.shape,
                                transformation=PositiveTransformation())
            q[v].set_prior(Normal(mean=mean, variance=variance, dtype=dtype))
    return q
