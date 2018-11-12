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


from ..common.exceptions import ModelSpecificationError
from ..components.variables import Variable, VariableType


def variables_to_UUID(variables):
    return [v.uuid if isinstance(v, Variable) else v for v in variables]


def realize_shape(shape, constants):
    rshape = []
    for s in shape:
        if isinstance(s, int):
            rshape.append(s)
        elif isinstance(s, Variable):
            if s.type == VariableType.CONSTANT:
                rshape.append(s.get_constant())
            elif s.type == VariableType.PARAMETER:
                rshape.append(constants[s.uuid])
    return tuple(rshape)


def discover_shape_constants(data_shapes, graphs):
    """
    The internal shape discovery from data and the shape inference for all the
    variables in the model and inference models.

    :param data_shapes: a dict of shapes of data
    :param graphs: a list of factor graphs of which variable shapes are
        searched.
    :returns: a dict of constants discovered from data shapes
    :rtype: {Variable: int}
    """
    shape_constants = {}
    variables = {}
    for g in graphs:
        variables.update(g.variables)
    for var_id, shape in data_shapes.items():
        def_shape = variables[var_id].shape
        for s1, s2 in zip(def_shape, shape):
            if isinstance(s1, int):
                if s1 != s2:
                    raise ModelSpecificationError("Variable ({}) shape mismatch between expected and found! s1 : {} s2 : {}".format(str(variables[var_id]),str(s1), str(s2)))
            elif isinstance(s1, Variable):
                shape_constants[s1] = s2
            else:
                raise ModelSpecificationError("The shape of a Variable should either an integer or a Variable, but encountered {}!".format(str(type(s1))))
    return shape_constants


def init_outcomes(inference_outcomes):
    if isinstance(inference_outcomes, list):
        updated_outcomes = inference_outcomes
    elif inference_outcomes is not None:
        updated_outcomes = [inference_outcomes]
    else:
        updated_outcomes = []
    return updated_outcomes
