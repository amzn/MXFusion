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


def broadcast_samples_dict(F, array_dict, num_samples=None):
    """
    Broadcast the shape of arrays in the provided dictionary. When the num_samples argument is given, all the sample dimesnions (the first dimension) of the arrays in the dictionary will be broadcasted to the size of num_samples. If the num_samples argument is not given, the sample dimensions of the arrays in the dictionary will be broadcasted to the maximum number of the sizes of the sample dimensions.

    :param F: the execution mode of MXNet.
    :type F: mxnet.ndarray or mxnet.symbol
    :param array_dict: the dictionary of arrays
    :type array_dict: {str: MXNet NDArray or Symbol}
    :param num_samples: (optional) the target size of the sample dimension
    :type num_samples: None or int
    """

    shape_dict = {k: v.shape for k, v in array_dict.items()}
    if num_samples is None:
        num_samples = max([s[0] for s in shape_dict.values()])

    if num_samples > 1:
        array_dict = {
            k: v if shape_dict[k] == num_samples
            else F.broadcast_to(v, (num_samples,) +
                                shape_dict[k][1:])
            for k, v in array_dict.items()}
    return array_dict


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
