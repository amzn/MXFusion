from .factor_graph import FactorGraph


class Posterior(FactorGraph):
    """
    A Posterior graph defined over an existing model.
    """

    def __init__(self, model, name=None):
        """
        Constructor.

        :param model:  The model which the posterior graph is defined over.
        :type model: Model
        """
        super(Posterior, self).__init__(name=name)
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
