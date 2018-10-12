from ..function_evaluation import FunctionEvaluation, FunctionEvaluationDecorator
from ...variables import Variable

class Operator(FunctionEvaluation):
    """
    Abstract Operator object for using MXNet operators in MXFusion space.
    Child classes implement the eval method with their operator and access necessary state through the properties dictionary.
    """
    def __init__(self, inputs, outputs, operator_name, properties=None, broadcastable=False):

        # TODO Add a flag for broadcastable

        input_names = [v[0] for v in inputs]
        output_names = [v[0] for v in outputs]
        self._properties = properties
        self.operator_name = operator_name
        self.broadcastable = broadcastable

        super(Operator, self).__init__(
            inputs, outputs, input_names, output_names, broadcastable=broadcastable)

    def replicate_self(self, attribute_map=None):
        replicant = super(Operator, self).replicate_self(attribute_map)
        replicant._properties = self._properties.copy()
        replicant.operator_name = self.operator_name
        return replicant

    @property
    def properties(self):
        return self._properties

class MXNetOperatorDecorator(object):

    def __init__(self, name, args, inputs, num_outputs=1, broadcastable=False):
        """
        :param name: The name of the operator to add.
        :type name: string
        :param args: The names of the arguments for the mxnet operator in order.
        :type args: list of strings
        :param inputs: The inputs to the MXNet operator that could have gradient's chained through them. I.E. the mx.nd.array or mx.sym.array parameters. This will be a subset of args (possibly the same set).
        :type inputs: list of strings
        :param num_outputs: How many output variables the operator produces. Defaults to 1.
        :type num_outputs: int
        """
        self.operator_name = name
        self.arg_names = args
        self.input_names = inputs
        self.property_names = [v for v in args if v not in inputs]
        self.num_outputs = num_outputs
        self.broadcastable = broadcastable

    def _parse_arguments(self, args, kwargs):
        arg_names = [v for v in self.arg_names if v not in kwargs]
        arguments = kwargs.copy()
        arguments.update({k: v for k, v in zip(arg_names, args)})
        return arguments

    def __call__(self, func):

        def create_operator(*args, **kwargs):
            all_args = self._parse_arguments(args, kwargs)

            class CustomOperator(Operator):

                @FunctionEvaluationDecorator()
                def eval(self, F, **input_kws):
                    input_kws.update(self.properties)
                    return func(F, **input_kws)

            op = CustomOperator(inputs=[(n, all_args[n]) for n in self.input_names if n in all_args],
                                  outputs=[('output_'+str(i), Variable()) for i in range(self.num_outputs)],
                                  operator_name=self.operator_name,
                                  properties={n: all_args[n] for n in self.property_names if n in all_args}
                                  )

            if self.num_outputs==1:
                return op.outputs[0][1]
            else:
                return tuple([op.outputs[i][1] for i in range(self.num_outputs)])
        return create_operator
