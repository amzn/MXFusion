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


class Posterior(FactorGraph):
    """
    A Posterior graph defined over an existing model.
    """

    def __init__(self, model, name=None, verbose=False):
        """
        Constructor.

        :param model:  The model which the posterior graph is defined over.
        :type model: Model
        """
        super(Posterior, self).__init__(name=name, verbose=verbose)
        self._model = model

    def __getattr__(self, name):
        if hasattr(self._model, name):
            v = getattr(self._model, name)

            replicant = v.replicate()
            setattr(self, name, replicant)
            return replicant
        else:
            raise AttributeError("''%s' object has no attribute '%s'" % (type(self), name))

    def __getitem__(self, item):
        if item in self.components:
            return self.components[item]
        elif item in self._model:
            object = self._model[item]
            replicant = object.replicate()
            if object.name is not None:
                setattr(self, object.name, replicant)
            else:
                replicant.graph = self.components_graph
            return replicant
        else:
            raise AttributeError("''%s' object has no item '%s'" % (type(self), item))

    def _replicate_class(self, **kwargs):
        """
        Return a new instance of the derived FactorGraph's class.
        """
        return Posterior(**kwargs)

    def clone(self, model, leaves=None):
        new_model = self._replicate_class(model=model, name=self.name, verbose=self._verbose)
        return self._clone(new_model, leaves)
