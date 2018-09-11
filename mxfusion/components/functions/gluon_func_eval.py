import mxnet as mx
from ...common.exceptions import InferenceError
from ..variables import VariableType
from .function_evaluation import FunctionEvaluation, FunctionEvaluationDecorator


class GluonFunctionEvaluation(FunctionEvaluation):
    """
    The evaluation of a function that is a wrapper of a MXNet Gluon block.

    :param function_wrapper: the MXFusion wrapper of the MXNet Gluon block that the function evaluation is associated with.
    :type function_wrapper: MXFusion.functions.MXFusionGluonFunction
    :param input_variables: the input arguments to the function.
    :type input_variables: {variable name : Variable}
    :param block_variables: the parameters in the Gluon block.
    :type block_variables: {variable name : Variable}
    :param num_outputs: the number of outputs of the function.
    :type num_outputs: int
    :param broadcastable: Whether the function supports broadcasting with the additional dimension for samples.
    :type: boolean
    """

    def __init__(self, function_wrapper, input_variables, block_variables,
                 output_variables, broadcastable=False):
        self.function_wrapper = function_wrapper
        self._input_variables_names = [k for k, _ in input_variables]
        self.block_variables_names = [k for k, _ in block_variables]

        inputs = block_variables + input_variables

        for _, bv in block_variables:
            if bv.type == VariableType.RANDVAR and broadcastable:
                # Broadcasting function evaluation can not be applied to the Gluon block with gluon block parameters as random variables.
                broadcastable = False

        input_names = [k for k, _ in inputs]
        output_names = [k for k, _ in output_variables]

        super(GluonFunctionEvaluation, self).__init__(
            inputs, output_variables,
            input_names, output_names, broadcastable=broadcastable)

    @property
    def _input_variables(self):
        return [(k, v) for k, v in self.inputs if k in self._input_variables_names]

    @property
    def block_variables(self):
        return [(k, v) for k, v in self.inputs if k in self.block_variables_names]

    def replicate_self(self, attribute_map=None):
        replicant = super(GluonFunctionEvaluation, self).replicate_self(attribute_map)
        replicant.function_wrapper = self.function_wrapper
        replicant.block_variables_names = self.block_variables_names
        replicant._input_variables_names = self._input_variables_names
        return replicant

    @FunctionEvaluationDecorator()
    def eval(self, F, **input_kws):
        """
        Invokes the MXNet Gluon block with the arguments passed in.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
        :param \**input_kws: the dict of inputs to the functions. The key in the dict should match with the name of inputs specified in the inputs
            of FunctionEvaluation.
        :type \**input_kws: {variable name: MXNet NDArray or MXNet Symbol}
        :returns: the return value of the function
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        inputs_func = [input_kws[k] for k, _ in self._input_variables]
        self._override_block_parameters(input_kws)
        return self.function_wrapper.block(*inputs_func)

    def _override_block_parameters(self, input_kws):
        """
        When a probabilistic distribution is defined for the parameters of a Gluon block (in ParameterDict), a special treatment is necessary
        because otherwise these parameters will be directly exposed to a gradient optimizer as free parameters.

        For each parameters of the Gluon bock with probabilistic distribution, this method dynamically sets its values as the outcome of
        upstream computation and ensure the correct gradient can be estimated via automatic differenciation.

        :param \**input_kws: the dict of inputs to the functions. The key in the dict should match with the name of inputs specified in the
            inputs of FunctionEvaluation.
        :type \**input_kws: {variable name: MXNet NDArray or MXNet Symbol}
        """
        for bn, bv in self.block_variables:
            if bv.type == VariableType.RANDVAR:
                if self.broadcastable:
                    raise InferenceError('Broadcasting function evaluation can not be applied to the Gluon block with gluon block parameters as random variables.')
                val = input_kws[bn]
                param = self.function_wrapper.block.collect_params()[bn]

                if isinstance(val, mx.ndarray.ndarray.NDArray):
                    ctx = val.context
                    ctx_list = param._ctx_map[ctx.device_typeid&1]
                    if ctx.device_id >= len(ctx_list) or ctx_list[ctx.device_id] is None:
                        raise Exception
                    dev_id = ctx_list[ctx.device_id]
                    param._data[dev_id] = val
                else:
                    param._var = val
