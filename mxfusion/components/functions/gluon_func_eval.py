from ..variables.variable import VariableType
from .function_evaluation import FunctionEvaluationWithParameters, \
    FunctionEvaluationDecorator


class GluonFunctionEvaluation(FunctionEvaluationWithParameters):
    """
    The evaluation of a function that is a wrapper of a MXNet Gluon block.

    :param func: the MXFusion wrapper of the MXNet Gluon block that the function evaluation is associated with.
    :type func: MXFusion.components.functions.MXFusionGluonFunction
    :param input_variables: the input arguments to the function.
    :type input_variables: {str : Variable}
    :param output_variables: the output variables of the function.
    :type output_variables: {str : Variable}
    :param broadcastable: Whether the function supports broadcasting with the additional dimension for samples.
    :type: boolean
    """
    def __init__(self, func, input_variables, output_variables,
                 broadcastable=False):
        super(GluonFunctionEvaluation, self).__init__(
            func=func, input_variables=input_variables,
            output_variables=output_variables, broadcastable=broadcastable
        )

    @property
    def _input_to_gluon_names(self):
        return [k for k, v in self.inputs if (not v.isInherited) or
                v.type != VariableType.PARAMETER]

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
        return self._func.eval(F, **inputs_func)
