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


from ..variables.variable import VariableType
from .function_evaluation import FunctionEvaluationWithParameters


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
