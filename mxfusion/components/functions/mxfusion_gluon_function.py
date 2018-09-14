from .gluon_func_eval import GluonFunctionEvaluation
from ..variables import Variable, VariableType
from mxnet.gluon import ParameterDict, Block, HybridBlock
from ...common.exceptions import ModelSpecificationError


class MXFusionGluonFunction(object):
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

        super(MXFusionGluonFunction, self).__init__()
        self.broadcastable = broadcastable
        self.block = block
        self._name = block.name
        self.num_outputs = num_outputs
        self._gluon_parameters = self._create_variables_from_gluon_block(block)

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

        # Copy the block variables so that the user can overwrite them with Random or Function Variables without
        # disturbing other FunctionEvaluations.
        # TODO: support binding (tying) a gluon parameter with a normal model parameter
        block_variables = self._gluon_parameters.copy()
        block_variables.update(kwargs)
        block_variables = [(k, v) for k, v in block_variables.items()]

        input_variables = [(self.block.name + "_input_" + str(i), variable) for i, variable in enumerate(args)]

        outputs = []
        for i in range(self.num_outputs):
            output = Variable()
            outputs.append((self.name + "_output_" + str(i), output))

        fe = GluonFunctionEvaluation(
            self, input_variables, block_variables, outputs,
            broadcastable=self.broadcastable)

        return tuple([v for _, v in fe.outputs]) if len(fe.outputs) > 1 else fe.outputs[0][1]

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

    def collect_internal_parameters(self):
        """
        Return the parameters of the MXNet Gluon block that have *not* been set a prior distribution.

        :returns: the parameters of the MXNet Gluon block without a prior distribution.
        :rtype: MXNet.gluon.ParameterDict
        """
        params = ParameterDict()
        gluon_params = self.block.collect_params()
        params.update({var_name: gluon_params[var_name] for var_name, var in self._gluon_parameters.items() if var.type == VariableType.PARAMETER})
        return params

    def collect_params(self):
        """
        Return a variable set / dict. Used for the function.collect_params.set_prior() functionality.
        """
        # TODO: implement VariableSet
        raise NotImplementedError

    @property
    def name(self):
        return self._name
