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


import re
import mxnet as mx
import numpy as np
from mxnet.ndarray.ndarray import NDArray
from mxnet.symbol.symbol import Symbol


def slice_axis(F, array, axis, indices):
    """
    Slice an array along a given axis by a list of indices.

    :param array: the array to be sliced.
    :type array: MXNet Array
    :param axis: the axis to be sliced.
    :type axis: int
    :param indices: the indices used in slicing
    :type indices: list or MXNet Array
    """
    assert F == mx.nd, "The slice_axis helper function only works on imperative mode, because fancy indexing only exists in NDArray API."
    if isinstance(indices, (list, tuple)):
        num_indices = len(indices)
    elif isinstance(indices, (NDArray, Symbol)):
        num_indices = indices.shape[0]
    if axis < 0:
        axis = len(array.shape) + axis
    if axis == 0:
        target_shape = (num_indices,) + array.shape[:1]
        array = F.reshape(array, shape=(array.shape[0], -1))[indices, :]
    elif axis == len(array.shape) - 1:
        target_shape = array.shape[:-1] + (num_indices,)
        array = F.reshape(array, shape=(-1, array.shape[-1]))[:, indices]
    else:
        target_shape = array.shape[:axis] + (num_indices,) + \
            array.shape[axis + 1:]

        def multiply(l):
            r = 1
            for i in l:
                r *= i
            return r

        size_before = multiply(array.shape[:axis])
        size_after = multiply(array.shape[axis + 1:])

        array = F.reshape(array, shape=(size_before, array.shape[axis],
                                        size_after))[:, indices, :]
    return F.reshape(array, shape=target_shape)


def rename_duplicate_names(names):
    """
    Given a list of names, rename the duplicated names by appending an integer at the end. This function returns a list of tuples each of
    which contains an index of the duplicate name and a new name.
    For example,
    ['a', 'b', 'a', 'a'] -> [(2, 'a0'), (3, 'a0')]
    ['a', 'b'] -> []

    :param names: the list of names
    :type names: [str]
    :returns: a list of tuples each of which contains an index of the duplicate name and a new name.
    :rtypes: [tuple]
    """
    all_names = set(names)
    if len(all_names) == len(names):
        return []
    cur_names = set()
    prog = re.compile(r'^(.*)(\d+)$')
    renames = []
    for i, n in enumerate(names):
        if n in cur_names:
            res = prog.match(n)
            if res is None or len(res.groups())==0:
                prefix = n
                count = 0
            else:
                prefix = res.groups()[0]
                count = int(res.groups()[1])+1
            while prefix + str(count) in all_names:
                count += 1
            renames.append((i, prefix + str(count)))
            all_names.add(prefix + str(count))
        else:
            cur_names.add(n)
    return renames


def parse_string_to_tuple(s):
    res = re.match(r'[\(\[](.*)[\)\]]', s)
    if res is None or len(res.groups())==0:
        raise Exception
    s_mid = res.groups()[0]
    return tuple(int(i) for i in s_mid.split(','))


def add_mxnet_params(block, var, grad_req='write'):
    """
    This function adds MXFusion variables to the MXNet Gluon block's parameters for use in computations via the params variable or kwargs
    parameter in hybrid_forward()

    :param block: MXNet Gluon Block to add the parameters into
    :param var: array of MXFusion variables to add to the block
    :param grad_req: gradient required for these variables?
    :return:
    """
    from ..components.variables import Variable, VariableType
    param_dict = block.params
    with block.name_scope():
        for v in var:
            if isinstance(v, Variable) and v.type == VariableType.CONSTANT:
                if v.is_scalar:
                    init = mx.initializer.Constant(v.val)
                    init.dumps()
                    parameter = param_dict.get(v.name, shape=v.shape, init=init, grad_req='null')
                    block.__setattr__(v.name, parameter)
            else:
                block.__setattr__(v.name, param_dict.get(v.name, shape=v.shape, init=mx.initializer.One(),
                grad_req=grad_req))


def create_variables_with_names(names):
    """
    Utility function to create a dictionary of Variables from a list of names.

    :param: names list of names
    """
    from ..components.variables import Variable
    return {n: Variable(n) for n in names}


def create_constant_from_values(var):
    """
    Utility function to create a constant variable from a raw value.

    :param: var the value of the constant
    """
    from ..components.variables import Constant

    if isinstance(var, (int, float, np.ndarray, mx.nd.NDArray)):
        return Constant(var)
    elif isinstance(var, (list, tuple)):
        return Constant(mx.nd.array(var))
    return var
