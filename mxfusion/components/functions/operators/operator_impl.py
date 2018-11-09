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


from . import MXNetOperatorDecorator

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
