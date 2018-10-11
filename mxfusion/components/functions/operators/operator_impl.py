from . import OperatorDecorator

""" Basic Arithmetic """
@OperatorDecorator(name='add', args=['x', 'y'], inputs=['x', 'y'])
def add(F, x, y):
    return F.add(x, y)

@OperatorDecorator(name='subtract', args=['x', 'y'], inputs=['x', 'y'])
def subtract(F, x, y):
    return F.subtract(x, y)

@OperatorDecorator(name='multiply', args=['x', 'y'], inputs=['x', 'y'])
def multiply(F, x, y):
    return F.multiply(x, y)

@OperatorDecorator(name='divide', args=['x', 'y'], inputs=['x', 'y'])
def divide(F, x, y):
    return F.divide(x, y)

@OperatorDecorator(name='power', args=['x', 'y'], inputs=['x', 'y'])
def power(F, x, y):
    return F.power(x, y)

""" Elementwise Operations """
@OperatorDecorator(name='square', args=['data'], inputs=['data'])
def square(F, data):
    return F.square(data)

@OperatorDecorator(name='exp', args=['data'], inputs=['data'])
def exp(F, data):
    return F.exp(data)

@OperatorDecorator(name='log', args=['data'], inputs=['data'])
def log(F, data):
    return F.log(data)

""" Aggregation """
@OperatorDecorator(name='sum', args=['data', 'axis'], inputs=['data'])
def sum(F, data, axis=None):
    return F.sum(data, axis)

@OperatorDecorator(name='mean', args=['data', 'axis'], inputs=['data'])
def mean(F, data, axis=None):
    return F.mean(data, axis)

@OperatorDecorator(name='prod', args=['data', 'axis'], inputs=['data'])
def prod(F, data, axis=None):
    return F.prod(data, axis)

""" Matrix Operations """
@OperatorDecorator(name='dot', args=['x', 'y'], inputs=['x', 'y'])
def dot(F, x, y):
    return F.linalg.gemm2(x, y)

# TODO Bring in the axis arguments once it's in the release version of MXNet
@OperatorDecorator(name='diag', args=['data', 'k', 'axis1', 'axis2'], inputs=['data'])
def diag(F, data, k=0, axis1=None, axis2=None):
    if axis1 is not None or axis2 is not None:
        raise Exception("axis1 and axis2 are not implemented yet.")
    return F.diag(data, k)

""" Matrix Manipulations """
@OperatorDecorator(name='reshape', args=['data', 'shape', 'reverse'], inputs=['data'])
def reshape(F, data, shape, reverse=False):
    return F.reshape(data=data, shape=shape, reverse=reverse)

@OperatorDecorator(name='transpose', args=['data', 'axes'], inputs=['data'])
def transpose(F, data, axes=None):
    axes = axes if axes is not None else []
    return F.transpose(data=data, axes=axes)
