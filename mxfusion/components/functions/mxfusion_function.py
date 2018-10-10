from abc import abstractmethod
from ...common.config import get_default_dtype
from ..variables import Variable
from .function_evaluation import FunctionEvaluationWithParameters


class MXFusionFunction(object):
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

    def __init__(self, func_name, dtype=None, broadcastable=False):
        super(MXFusionFunction, self).__init__()
        self.broadcastable = broadcastable
        self._func_name = func_name
        self.dtype = get_default_dtype() if dtype is None else dtype

    @abstractmethod
    def eval(self, F, **input_kws):
        """
        The method handling the execution of the function.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
        :param **input_kws: the dict of inputs to the functions. The key in the
        dict should match with the name of inputs specified in the inputs of
        FunctionEvaluation.
        :type **input_kws: {variable name: MXNet NDArray or MXNet Symbol}
        :returns: the return value of the function
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        raise NotImplementedError

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
        given_args = self._parse_arguments(args, kwargs)
        input_variables = [(k, given_args[k]) for k in self.input_names if k in
                           given_args]
        output_variables = [(k, Variable()) for k in self.output_names]

        fe = FunctionEvaluationWithParameters(
            func=self, input_variables=input_variables,
            output_variables=output_variables,
            broadcastable=self.broadcastable)

        return tuple([v for _, v in fe.outputs]) if len(fe.outputs) > 1 else \
            fe.outputs[0][1]

    @property
    def parameters(self):
        """
        All the kernel parameters including the kernel parameters that belongs to the sub-kernels. The keys of the returned dictionary are the name of
        the kernel parameters with a prefix (the name of the kernel plus '_') and the values are the corresponding variables.

        :return: a dictionary of all the kernel parameters, in which the keys are the name of individual parameters, including the kernel in front,
            and the values are the corresponding Variables.
        :rtype: {str: Variable}
        """
        raise NotImplementedError

    @property
    def input_names(self):
        raise NotImplementedError

    @property
    def output_names(self):
        raise NotImplementedError

    @property
    def parameter_names(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._func_name

    @name.setter
    def name(self, name):
        self._func_name = name

    def _parse_arguments(self, args, kwargs):
        arg_names = [v for v in self.input_names if v not in kwargs]
        arguments = kwargs.copy()
        arguments.update({k: v for k, v in zip(arg_names, args)})
        return arguments

    def replicate_self(self, attribute_map=None):
        """
        This functions is a copy constructor for the object.
        In order to perform copy construction we first call ``__new__()`` on the class which creates a blank object.
        We then initialize that object using the method's standard init procedures, and do any extra copying of attributes.

        Replicates this Factor, using new inputs, outputs, and a new uuid.
        Used during model replication to functionally replicate a factor into a new graph.

        :param inputs: new input variables of the factor.
        :type inputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
        :param outputs: new output variables of the factor.
        :type outputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
        """
        replicant = self.__class__.__new__(self.__class__)

        MXFusionFunction.__init__(
            replicant, func_name=self.name, dtype=self.dtype,
            broadcastable=self.broadcastable)
        return replicant
