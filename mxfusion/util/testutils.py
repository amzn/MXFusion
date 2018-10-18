import mxnet as mx
import numpy as np
import mxfusion as mf
import mxnet.gluon.nn as nn
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxnet.gluon import HybridBlock
from ..components.distributions.random_gen import RandomGenerator
from ..components.variables import add_sample_dimension

def prepare_mxnet_array(array, is_sampled_array, dtype):
    a_mx = mx.nd.array(array, dtype=dtype)
    if not is_sampled_array:
        a_mx = add_sample_dimension(mx.nd, a_mx)
    return a_mx


def numpy_array_reshape(var, isSamples, n_dim):
    """
    Reshape a numpy to a give dimensionality by adding dimensions with size one in front (broadcasting). If *isSamples* is true, keep the first dimension.

    :param var: the variable to be reshaped.
    :type var: numpy.ndarray
    :param isSamples: whether the variable is a sampled variable with the first dimension being the number of samples.
    :type isSamples: boolean
    :param n_dim: the dimensionality that the variable is reshaped to.
    :type n_dim: int
    :returns: the reshaped numpy array
    :rtype: numpy.ndarray
    """
    if len(var.shape) < n_dim:
        if isSamples:
            var_reshape = var.reshape(
                *((var.shape[0],) + (1,) * (n_dim - len(var.shape)) +
                  var.shape[1:]))
        else:
            var_reshape = var.reshape(*((1,) * (n_dim - len(var.shape)) +
                                      var.shape))
    else:
        var_reshape = var
    return var_reshape


class MockMXNetRandomGenerator(RandomGenerator):
    """
    The MXNet pseudo-random number generator.
    """

    def __init__(self, samples):
        self._samples = samples

    def sample_normal(self, loc=0, scale=1, shape=None, dtype=None, out=None,
                      ctx=None, F=None):
        if shape is None:
            shape = (1,)
        res = mx.nd.reshape(self._samples[:np.prod(shape)], shape=shape)
        if out is not None:
            out[:] = res
        return res

    def sample_multinomial(self, data, shape=None, get_prob=True, dtype='int32',
                           F=None):
        return mx.nd.reshape(self._samples[:np.prod(data.shape[:-1])], shape=data.shape[:-1])

    def sample_gamma(self, alpha=1, beta=1, shape=None, dtype=None, out=None, ctx=None, F=None):
        if shape is None:
            shape = (1,)
        res = mx.nd.reshape(self._samples[:np.prod(shape)], shape=shape)
        if out is not None:
            out[:] = res
        return res


def make_net():
    D=100
    net = nn.HybridSequential(prefix='testnet0_')
    with net.name_scope():
        net.add(nn.Dense(D, activation="tanh"))
        net.add(nn.Dense(D, activation="tanh"))
        net.add(nn.Dense(1, flatten=True))
    net.initialize(mx.init.Xavier(magnitude=3))
    return net


def make_basic_model(finalize=True, verbose=True):
    net = make_net()
    m = mf.models.Model(verbose=verbose)
    m.N = mf.components.Variable()
    m.x = mf.components.Variable(shape=(m.N,))
    m.v = mf.components.Variable(shape=(1,), transformation=PositiveTransformation())
    m.r = mf.components.Variable(shape=(m.N,))
    m.y = mf.components.distributions.Normal.define_variable(mean=m.r, variance=m.v, shape=(m.N,))
    return m

def make_bnn_model(finalize=True, verbose=True):
    net = make_net()
    m = mf.models.Model(verbose=verbose)
    m.N = mf.components.Variable()
    m.f = mf.components.MXFusionGluonFunction(net, num_outputs=1)
    m.x = mf.components.Variable(shape=(m.N,))
    m.v = mf.components.Variable(shape=(1,), transformation=PositiveTransformation())
    m.prior_variance = mf.components.Variable(shape=(1,), transformation=PositiveTransformation())
    m.r = m.f(m.x)
    for _, v in m.r.factor.parameters.items():
        v.set_prior(mf.components.distributions.Normal(mean=mx.nd.array([0]), variance=m.prior_variance))
    m.y = mf.components.distributions.Normal.define_variable(mean=m.r, variance=m.v, shape=(m.N,))
    return m


class DotProduct(HybridBlock):
    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.dot(x, args[0])

class TestBlock(mx.gluon.HybridBlock):
    """
    Block with standard functions for initializing and running an MXNet Gluon block for unit testing.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(TestBlock, self).__init__()
        with self.name_scope():
            self.var1 = self.params.get('var1', shape=(2,))
            self.var2 = self.params.get('var2', shape=(2,))

    def hybrid_forward(self, F, x, var1, var2):
        """
        Simple test function to enable MXFusion tests that use a Gluon block.
        :param F: MXNet computation type <mx.sym, mx.nd>
        :param x: MXNet dummy variable
        :param var1: MXNet dummy variable
        :param var2: MXNet dummy variable
        """

        return x + var1 + var2
