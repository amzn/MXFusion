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
