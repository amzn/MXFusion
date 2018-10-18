from abc import abstractmethod
from ...common.config import get_default_dtype
from ..variables import Variable
from .function_evaluation import FunctionEvaluationWithParameters


class MXFusionFunction(object):
    """
    The base class for the functions in MXFusion.

    :param func_name: The name of the function
    :type func_name: str
    :param dtype: the data type of float point numbers used in the Gluon block.
    :type dtype: numpy.float32 or numpy.float64
    :param broadcastable: Whether the function supports broadcasting with the
    additional dimension for samples.
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
        The evaluation of the function in a model defition. It takes a list of
        arguments in the type of MXFusion Variable and returns the output
        variables.

        :param args: the positional arguments that the function takes.
        :type args: [Variable]
        :param kwargs: the keyword arguments that the function takes.
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
        The parameters in the format of MXFusion Variable that are associated
        with the function. These parameters are automatically included as the
        inputs in the resulting FunctionEvaluation object with the need of
        explicitly specification when calling the __call__ function.

        :return: a dictionary of all the parameters, in which the keys are the
        name of individual parameters and the values are the corresponding
        Variables.
        :rtype: {str: Variable}
        """
        raise NotImplementedError

    @property
    def input_names(self):
        """
        The names of all the inputs that the function takes including the function parameters
        """
        raise NotImplementedError

    @property
    def output_names(self):
        """
        The names of all the outputs of the function
        """
        raise NotImplementedError

    @property
    def parameter_names(self):
        """
        The names of all the function parameters.
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        The name of the function
        """
        return self._func_name

    @name.setter
    def name(self, name):
        """
        The setter of the name of the function
        """
        self._func_name = name

    def _parse_arguments(self, args, kwargs):
        """
        Parse the input positional and keyword arguments for the __call__ function.
        """
        arg_names = [v for v in self.input_names if v not in kwargs]
        arguments = kwargs.copy()
        arguments.update({k: v for k, v in zip(arg_names, args)})
        return arguments

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for the fuction.
        """
        replicant = self.__class__.__new__(self.__class__)

        MXFusionFunction.__init__(
            replicant, func_name=self.name, dtype=self.dtype,
            broadcastable=self.broadcastable)
        return replicant
