from abc import abstractmethod
from ..factor import Factor
from ..variables import is_sampled_array, get_num_samples, as_samples
from ..variables import VariableType


class FunctionEvaluationDecorator(object):
    """
    The decorator for the eval function in FunctionEvaluation
    """
    def __call__(self, func):

        func_reshaped = self._wrap_eval_with_reshape(func)
        func_variables = self._wrap_eval_with_variables(func_reshaped)
        return func_variables

    def _wrap_eval_with_variables(self, func):
        def eval_RT_variableset(self, F, variables, always_return_tuple=False):
            """
            The method handling the execution of the function. The inputs arguments of the function are fetched from the *variables* argument according to their UUIDs.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param variables: the set of MXNet arrays that holds the values of variables at runtime.
            :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
            :param always_return_tuple: Whether return a tuple even if there is
            only one variables in outputs.
            :type always_return_tuple: boolean
            :returns: the return value of the function
            :rtypes: MXNet NDArray or MXNet Symbol
            """
            args = {name: variables[var.uuid] for name, var in self.inputs
                    if not var.isInherited or var.type == VariableType.RANDVAR}
            return func(
                self, F=F, always_return_tuple=always_return_tuple, **args)
        return eval_RT_variableset

    def _wrap_eval_with_reshape(self, func):
        def eval_RT(self, F, always_return_tuple=False, **input_kws):
            """
            The method handling the execution of the function with RTVariable
            as its input arguments and return values.
            """
            has_samples = any([is_sampled_array(F, v) for v in input_kws.values()])
            if not has_samples:
                # If none of the inputs are samples, directly evaluate the function
                nSamples = 0
                results = func(self, F=F, **{n: v[0] for n, v in input_kws.items()})
                if isinstance(results, (list, tuple)):
                    results = [F.expand_dims(r, axis=0) for r in results]
                else:
                    results = F.expand_dims(results, axis=0)
            else:
                nSamples = max([get_num_samples(F, v) for v in input_kws.values()])
                if self.broadcastable:
                    # If some of the inputs are samples and the function is
                    # broadcastable, evaluate the function with the inputs that are
                    # broadcasted to the right shape.
                    results = func(self, F=F, **{n: as_samples(F, v, nSamples)
                                                 for n, v in input_kws.items()})
                else:
                    # If some of the inputs are samples and the function is *not*
                    # broadcastable, evaluate the function with each set of samples
                    # and concatenate the output variables.
                    results = []
                    for sample_idx in range(nSamples):
                        r = func(
                            self, F=F, **{
                                n: v[sample_idx] if is_sampled_array(F, v) else
                                v[0] for n, v in input_kws.items()})
                        if isinstance(r, (list, tuple)):
                            r = [F.expand_dims(i, axis=0) for i in r]
                        else:
                            r = F.expand_dims(r, axis=0)
                        results.append(r)
                    if isinstance(results[0], (list, tuple)):
                        # if the function has multiple output variables.
                        results = [F.concat([r[i] for r in results], dim=0) for
                                   i in range(len(results[0]))]
                    else:
                        results = F.concat(*results, dim=0)
            results = results if isinstance(results, (list, tuple)) \
                else [results]
            if len(results) == 1 and not always_return_tuple:
                results = results[0]
            return results
        return eval_RT


class FunctionEvaluation(Factor):
    """
    The evaluation of a function with specified input variables.

    :param inputs: the input variables to the function.
    :type inputs:  {variable name : Variable}
    :param outputs: the output variables to the function.
    :type outputs: {variable name : Variable}
    :param broadcastable: Whether the function supports broadcasting with the additional dimension for samples.
    :type: boolean
    """
    def __init__(self, inputs, outputs, input_names, output_names,
                 broadcastable=False):
        self.broadcastable = broadcastable
        super(FunctionEvaluation, self).__init__(
            inputs=inputs, outputs=outputs, input_names=input_names,
            output_names=output_names)

    def replicate_self(self, attribute_map=None):
        replicant = super(FunctionEvaluation, self).replicate_self(attribute_map)
        replicant.broadcastable = self.broadcastable
        return replicant

    @abstractmethod
    @FunctionEvaluationDecorator()
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


class FunctionEvaluationWithParameters(FunctionEvaluation):
    """
    The evaluation of a function with internal function parameters.

    :param func: the function that this evaluation is generated from
    :param inputs: MXFusion.components.functions.MXFusionFunction
    :type inputs:  {str : Variable}
    :param outputs: the output variables of the function.
    :type outputs: {str : Variable}
    :param broadcastable: Whether the function supports broadcasting with the additional dimension for samples.
    :type: boolean
    """
    def __init__(self, func, input_variables, output_variables,
                 broadcastable=False):
        input_variable_names = set([k for k, _ in input_variables])
        inputs = input_variables + list(
            {k: v for k, v in func.parameters.items() if k not in
             input_variable_names}.items())
        input_names = [k for k, _ in inputs]
        output_names = [k for k, _ in output_variables]
        super(FunctionEvaluationWithParameters, self).__init__(
            inputs=inputs, outputs=output_variables, input_names=input_names,
            output_names=output_names, broadcastable=broadcastable
        )
        self._func = func

    def replicate_self(self, attribute_map=None):
        replicant = super(
            FunctionEvaluationWithParameters,
            self).replicate_self(attribute_map)
        replicant._func = self._func.replicate_self(attribute_map)
        return replicant

    @property
    def parameters(self):
        return self._func.parameters

    @property
    def function(self):
        return self._func

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
        return self._func.eval(F, **input_kws)
