import mxnet as mx
from copy import copy
from .gluon_func_eval import GluonFunctionEvaluation
from .mxfusion_function import MXFusionFunction
from ..variables import Variable, VariableType
from mxnet.gluon import ParameterDict, Block, HybridBlock
from ...common.exceptions import ModelSpecificationError, InferenceError


class MXFusionGluonFunction(MXFusionFunction):
    """
    The wrapper of a MXNet Gluon block in MXFusion. It automatically fetches all the Gluon parameters in its ParameterDict. When this function
    wrapper is called in Model definition, it returns a factor corresponding to the function evaluation.

    :param block: The MXNet Gluon block to be wrapped.
    :type block: mxnet.gluon.Blockk or mxnet.gluon.HybridBlock
    :param num_outputs: The number of output variables of the Gluon block.
    :type num_outputs: int
    :param dtype: the data type of float point numbers used in the Gluon block.
    :type dtype: numpy.float32 or numpy.float64
    :param broadcastable: Whether the function supports broadcasting with the additional dimension for samples.
    :type broadcastable: boolean
    """

    def __init__(self, block, num_outputs, dtype=None, broadcastable=False):

        if not isinstance(block, (Block, HybridBlock)):
            raise ModelSpecificationError('The block argument must be of type Block or HybridBlock from MXNet Gluon.')

        super(MXFusionGluonFunction, self).__init__(
            func_name=block.name, dtype=dtype, broadcastable=broadcastable)
        self._gluon_block = block
        self.num_outputs = num_outputs
        self._gluon_parameters = self._create_variables_from_gluon_block(block)
        self._input_names = None
        self._input_variable_names = None
        self._output_names = [self.name + "_output_" + str(i) for i in
                              range(self.num_outputs)]
        self._gluon_parameter_names = sorted(self._gluon_parameters.keys())

    @property
    def gluon_block(self):
        return self._gluon_block

    @property
    def input_names(self):
        """
        The names of all the inputs that the function takes including the function parameters
        """
        return self._input_names

    @property
    def output_names(self):
        """
        The names of all the outputs of the function
        """
        return self._output_names

    @property
    def parameter_names(self):
        """
        The names of all the function parameters.
        """
        return self._gluon_parameter_names

    @property
    def parameters(self):
        """
        The parameters in the format of MXFusion Variable that are associated
        with the function. These parameters are automatically included as the
        inputs in the resulting FunctionEvaluation object with the need of
        explicitly specification when calling the __call__ function.

        :return: a dictionary of all the parameters, in which the keys are the
        name of individual parameters and the values are the corresponding
        Variables.
        :rtype: {str: Variable}
        """
        return self._gluon_parameters

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
        inputs_func = [input_kws[k] for k in self._input_variable_names]
        self._override_block_parameters(input_kws)
        return self._gluon_block(*inputs_func)

    def __call__(self, *args, **kwargs):
        """
        Bind the wrapped Gluon block with the input arguments. This is called during model specification.

        :param args: the positional arguments that the Gluon block takes.
        :type args: [Variable]
        :param kwargs: Internal block variables the user would like to overwrite with Random or Function Variables.
        :type kwargs: {name : Variable}
        :returns: The output variables of the FunctionEvaluation with the specified inputs.
        :rtype: A tuple of output Variables if >1 or a single output Variable if 1
        """
        self._input_variable_names = [self.name + "_input_" + str(i) for i in
                                      range(len(args))]
        self._input_names = self._input_variable_names + \
            self.parameter_names

        broadcastable = self.broadcastable
        for bv in kwargs.values():
            if bv.type != VariableType.PARAMETER and self.broadcastable:
                # Broadcasting function evaluation can not be applied to the Gluon block with gluon block parameters as random variables.
                broadcastable = False
                break

        given_args = self._parse_arguments(args, kwargs)
        input_variables = [(k, given_args[k]) for k in self.input_names if k in
                           given_args]
        output_variables = [(k, Variable()) for k in self.output_names]

        fe = GluonFunctionEvaluation(
            func=self, input_variables=input_variables,
            output_variables=output_variables,
            broadcastable=broadcastable)

        return tuple([v for _, v in fe.outputs]) if len(fe.outputs) > 1 else \
            fe.outputs[0][1]

    def _create_variables_from_gluon_block(self, block):
        """
        Create a list of Parameter type variables from a MXNet Gluon block's parameters.
        One variable per parameter.

        :param block: The Block to create variables from.
        :rtype: {'block_variable.name' : block_variable}
        """
        params = block.collect_params()
        vs = {}
        for param in params.values():
            v = Variable(isInherited=True, shape=param.shape)
            v.inherited_name = param.name
            vs[v.inherited_name] = v
        return vs

    def collect_gluon_parameters(self):
        """
        Return the parameters of the MXNet Gluon block that have *not* been set a prior distribution.

        :returns: the parameters of the MXNet Gluon block without a prior distribution.
        :rtype: MXNet.gluon.ParameterDict
        """
        params = ParameterDict()
        gluon_params = self._gluon_block.collect_params()
        params.update({var_name: gluon_params[var_name] for var_name, var in self._gluon_parameters.items() if var.type == VariableType.PARAMETER})
        return params

    def collect_params(self):
        """
        Return a variable set / dict. Used for the function.collect_params.set_prior() functionality.
        """
        # TODO: implement VariableSet
        raise NotImplementedError

    def _override_block_parameters(self, input_kws):
        """
        When a probabilistic distribution is defined for the parameters of a Gluon block (in ParameterDict), a special treatment is necessary
        because otherwise these parameters will be directly exposed to a gradient optimizer as free parameters.

        For each parameters of the Gluon bock with probabilistic distribution, this method dynamically sets its values as the outcome of
        upstream computation and ensure the correct gradient can be estimated via automatic differenciation.

        :param **input_kws: the dict of inputs to the functions. The key in the dict should match with the name of inputs specified in the
            inputs of FunctionEvaluation.
        :type **input_kws: {variable name: MXNet NDArray or MXNet Symbol}
        """
        for bn in self.parameter_names:
            if bn in input_kws:
                val = input_kws[bn]
                param = self._gluon_block.collect_params()[bn]

                if isinstance(val, mx.ndarray.ndarray.NDArray):
                    ctx = val.context
                    ctx_list = param._ctx_map[ctx.device_typeid&1]
                    if ctx.device_id >= len(ctx_list) or ctx_list[ctx.device_id] is None:
                        raise Exception
                    dev_id = ctx_list[ctx.device_id]
                    param._data[dev_id] = val
                else:
                    param._var = val

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for the fuction.
        """
        replicant = super(
            MXFusionGluonFunction, self).replicate_self(attribute_map)
        replicant._gluon_block = self._gluon_block
        replicant.num_outputs = self.num_outputs
        replicant._gluon_parameters = {
            k: v.replicate_self(attribute_map) for k, v in
            self._gluon_parameters.items()}
        replicant._input_names = copy(self._input_names)
        replicant._input_variable_names = copy(self._input_variable_names)
        replicant._output_names = copy(self._output_names)
        replicant._gluon_parameter_names = \
            sorted(replicant._gluon_parameters.keys())
        return replicant
