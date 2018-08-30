from .factor import Factor


class Module(Factor):
    """
    Modules are a combined Model/Posterior + Inference algorithm object in MXFusion. They act as Factors and are defined as such during model definition, producing Random Variables like a typical Distribution. During inference, instead of the Module having a closed form solution, the outside inference algorithm will call the module's internal inference algorithm to perform modular inference.
    """

    def __init__(self, graph, inputs, outputs, input_names, output_names):
        self._graph = graph

        super(Module, self).__init__(inputs=inputs, outputs=outputs, input_names=input_names, output_names=output_names)

    def compute_log_prob(self, F, targets, conditionals=None, constants=None):
        raise NotImplementedError

    def draw_samples(self, F, num_samples=1, targets=None, conditionals=None,
                     constants=None):
        raise NotImplementedError
