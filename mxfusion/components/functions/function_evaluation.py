# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================


from abc import abstractmethod
from ..factor import Factor
from ..variables import array_has_samples, get_num_samples
from ..variables import VariableType
from ...util.inference import broadcast_samples_dict


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
        replicant = super(
            FunctionEvaluation, self).replicate_self(attribute_map)
        replicant.broadcastable = self.broadcastable
        return replicant

    def eval(self, F, variables, always_return_tuple=False):
        """
        Evaluate the function with the pre-specified input arguments in the model defintion. All the input arguments are automatically collected from a dictionary of variables according to the UUIDs of the input arguments.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :param variables: the set of variables where the dependent variables are collected from.
        :type variables: {str(UUID): MXNet NDArray or Symbol}
        :param always_return_tuple: whether to always return the function outcome in a tuple, even if there is only one output variable. This makes programming easy, as the downstream code can consistently expect a tuple.
        :type always_return_tuple: boolean
        :returns: the outcome of the function evaluation
        :rtypes: MXNet NDArray or MXNet Symbol or [MXNet NDArray or MXNet Symbol]
        """
        kwargs = {name: variables[var.uuid] for name, var in self.inputs
                  if not var.isInherited or var.type == VariableType.RANDVAR}
        if self.broadcastable:
            # If some of the inputs are samples and the function is
            # broadcastable, evaluate the function with the inputs that are
            # broadcasted to the right shape.
            kwargs = broadcast_samples_dict(F, kwargs)
            results = self.eval_impl(F=F, **kwargs)
            results = results if isinstance(results, (list, tuple)) \
                else [results]
        else:
            # If some of the inputs are samples and the function is *not*
            # broadcastable, evaluate the function with each set of samples
            # and concatenate the output variables.
            nSamples = max([get_num_samples(F, v) for v in kwargs.values()])

            results = None
            for sample_idx in range(nSamples):
                r = self.eval_impl(F=F, **{
                        n: v[sample_idx] if array_has_samples(F, v) else v[0]
                        for n, v in kwargs.items()})
                if isinstance(r, (list, tuple)):
                    r = [F.expand_dims(r_i, axis=0) for r_i in r]
                else:
                    r = [F.expand_dims(r, axis=0)]
                if results is None:
                    results = [[r_i] for r_i in r]
                else:
                    for r_list, r_i in zip(results, r):
                        r_list.append(r_i)
            if nSamples == 1:
                results = [r[0] for r in results]
            else:
                results = [F.concat(*r, dim=0) for r in results]
        if len(results) == 1 and not always_return_tuple:
            results = results[0]
        return results

    @abstractmethod
    def eval_impl(self, F, **input_kws):
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

    def eval_impl(self, F, **input_kws):
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
