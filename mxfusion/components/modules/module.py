from ..factor import Factor


class Module(Factor):
    """
    Modules are a combined Model/Posterior + Inference algorithm object in MXFusion. They act as Factors and are defined as such during model
    definition, producing Random Variables like a typical Distribution. During inference, instead of the Module having a closed form solution,
    the outside inference algorithm will call the module's internal inference algorithm to perform modular inference.
    """

    def __init__(self, model, inferences):
        """
        Constructor.

        :param model: The model for this module.
        :param inferences: Inferences for this module.
        """
        self._model = model
        self._inferences = inferences

    def create_variable(self):
        """
        Create a variable drawn from this module for use during model building.
        """
        raise NotImplementedError
