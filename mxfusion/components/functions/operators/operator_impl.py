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


import mxnet as mx
from . import MXNetOperatorDecorator
from .operators import Operator
from ...variables import Variable
from ....util.inference import realize_shape
from ....common.exceptions import InferenceError


""" Basic Arithmetic """
@MXNetOperatorDecorator(name='add', args=['x', 'y'], inputs=['x', 'y'])
def add(F, x, y):
    return F.add(x, y)

@MXNetOperatorDecorator(name='subtract', args=['x', 'y'], inputs=['x', 'y'])
def subtract(F, x, y):
    return F.subtract(x, y)

@MXNetOperatorDecorator(name='multiply', args=['x', 'y'], inputs=['x', 'y'])
def multiply(F, x, y):
    return F.multiply(x, y)

@MXNetOperatorDecorator(name='divide', args=['x', 'y'], inputs=['x', 'y'])
def divide(F, x, y):
    return F.divide(x, y)

@MXNetOperatorDecorator(name='power', args=['x', 'y'], inputs=['x', 'y'])
def power(F, x, y):
    return F.power(x, y)

""" Elementwise Operations """
@MXNetOperatorDecorator(name='square', args=['data'], inputs=['data'])
def square(F, data):
    return F.square(data)

@MXNetOperatorDecorator(name='exp', args=['data'], inputs=['data'])
def exp(F, data):
    return F.exp(data)

@MXNetOperatorDecorator(name='log', args=['data'], inputs=['data'])
def log(F, data):
    return F.log(data)

""" Aggregation """
@MXNetOperatorDecorator(name='sum', args=['data', 'axis'], inputs=['data'])
def sum(F, data, axis=None):
    return F.sum(data, axis)

@MXNetOperatorDecorator(name='mean', args=['data', 'axis'], inputs=['data'])
def mean(F, data, axis=None):
    return F.mean(data, axis)

@MXNetOperatorDecorator(name='prod', args=['data', 'axis'], inputs=['data'])
def prod(F, data, axis=None):
    return F.prod(data, axis)

""" Matrix Operations """
@MXNetOperatorDecorator(name='dot', args=['x', 'y'], inputs=['x', 'y'])
def dot(F, x, y):
    return F.linalg.gemm2(x, y)

# TODO Bring in the axis arguments once it's in the release version of MXNet
@MXNetOperatorDecorator(name='diag', args=['data', 'k', 'axis1', 'axis2'], inputs=['data'])
def diag(F, data, k=0, axis1=None, axis2=None):
    if axis1 is not None or axis2 is not None:
        raise Exception("axis1 and axis2 are not implemented yet.")
    return F.diag(data, k)

""" Matrix Manipulations """
@MXNetOperatorDecorator(name='reshape', args=['data', 'shape', 'reverse'], inputs=['data'])
def reshape(F, data, shape, reverse=False):
    return F.reshape(data=data, shape=shape, reverse=reverse)

@MXNetOperatorDecorator(name='transpose', args=['data', 'axes'], inputs=['data'])
def transpose(F, data, axes=None):
    axes = axes if axes is not None else []
    return F.transpose(data=data, axes=axes)


"""Special Operators"""


def broadcast_to(data, shape):
    """
    This operator broadcast a variable to a target shape. The broadcasting rule is the same as [the numpy broadcasting rule](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html). See the following example:

    ```python
    m.x = Gaussian.define_variable(mean=broadcast_to(array([0]), (2,)),
                                   variance=broadcast_to(array([1]), (2,))),
                                   shape=(2,))
    ```

    :param data: the variable to be broadcasted
    :type data: Variable
    :param shape: the shape of which the variable will be broadcasted to
    :type shape: tuple of int or Variable
    """
    class BroadcastToOperator(Operator):

        def __init__(self, data, shape):
            super(BroadcastToOperator, self).__init__(
                inputs=[('data', data)],
                outputs=[('output_0', Variable(shape=None))],
                operator_name='broadcast_to', properties={'shape': shape},
                broadcastable=True)

        def eval(self, F, variables, always_return_tuple=False):
            target_shape = realize_shape(self.properties['shape'], variables)
            data = variables[self.inputs[0][1].uuid]
            if F is mx.ndarray:
                source_shape = data.shape
            elif F is mx.symbol:
                raise NotImplementedError('Symbolic mode to be supported!')
            else:
                raise InferenceError('Unknown MXNet Mode '+str(F))
            n_target_dim = len(target_shape)
            n_source_dim = len(source_shape)

            if n_target_dim + 1 - n_source_dim > 0:
                t_shape = (source_shape[0],) + \
                    (1,) * (n_target_dim + 1 - n_source_dim) + source_shape[1:]
                data = F.reshape(data, shape=t_shape)
            shape = (source_shape[0],) + target_shape
            res = F.broadcast_to(data, shape=shape)
            if always_return_tuple:
                res = (res,)
            return res

    op = BroadcastToOperator(data=data, shape=shape)
    return op.outputs[0][1]
