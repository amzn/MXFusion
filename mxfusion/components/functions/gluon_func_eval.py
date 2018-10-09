from .function_evaluation import FunctionEvaluationWithParameters, \
    FunctionEvaluationDecorator


class GluonFunctionEvaluation(FunctionEvaluationWithParameters):
    """
    The evaluation of a function that is a wrapper of a MXNet Gluon block.

    :param function_wrapper: the MXFusion wrapper of the MXNet Gluon block that the function evaluation is associated with.
    :type function_wrapper: MXFusion.functions.MXFusionGluonFunction
    :param input_variables: the input arguments to the function.
    :type input_variables: {variable name : Variable}
    :param parameters.items(): the parameters in the Gluon block.
    :type parameters.items(): {variable name : Variable}
    :param num_outputs: the number of outputs of the function.
    :type num_outputs: int
    :param broadcastable: Whether the function supports broadcasting with the additional dimension for samples.
    :type: boolean
    """
    def __init__(self, func, input_variables, output_variables,
                 broadcastable=False):
        super(GluonFunctionEvaluation, self).__init__(
            func=func, input_variables=input_variables,
            output_variables=output_variables, broadcastable=broadcastable
        )
        self._input_to_gluon_names = [k for k, v in self.inputs if not
                                      v.isInherited]

    def replicate_self(self, attribute_map=None):
        replicant = super(
            GluonFunctionEvaluation, self).replicate_self(attribute_map)
        replicant._input_to_gluon_names = self._input_to_gluon_names
        return replicant

    @FunctionEvaluationDecorator()
    def eval(self, F, **input_kws):
        """
        Invokes the MXNet Gluon block with the arguments passed in.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
        :param **input_kws: the dict of inputs to the functions. The key in the dict should match with the name of inputs specified in the inputs
            of FunctionEvaluation.
        :type **input_kws: {variable name: MXNet NDArray or MXNet Symbol}
        :returns: the return value of the function
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        inputs_func = {k: input_kws[k] for k in self._input_to_gluon_names}
        return self._func.eval(F, self.broadcastable, **inputs_func)
