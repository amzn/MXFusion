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


from .factor_graph import FactorGraph
from ..components import VariableType


class Model(FactorGraph):
    """
    The model defines a factor graph over a set of variables for use in inference.
    """

    def __init__(self, name=None, verbose=False):
        """
        Model object constructor.

        :param name: optional parameter to name the model for easier reference.
        """
        super(Model, self).__init__(name=name, verbose=verbose)

    def get_latent_variables(self, observed):
        """
        Get the latent variables of the model.

        :param observed: a list of observed variables.
        :type observed: [UUID]
        :returns: the list of latent variables.
        :rtype: [Variable]
        """
        return [v for v in self.variables.values() if v.type == VariableType.RANDVAR and v.uuid not in observed]

    def _replicate_class(self, **kwargs):
        """
        Returns a new instance of the derived FactorGraph's class.
        """
        return Model(**kwargs)
