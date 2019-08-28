import warnings
warnings.filterwarnings('ignore')
import os
from mxfusion import Variable, Model
from mxfusion.components.variables import PositiveTransformation
from mxfusion.components.distributions import Normal
from mxfusion.common import config
config.DEFAULT_DTYPE = 'float64'
import mxnet.gluon.nn as nn
from mxfusion.components.functions import MXFusionGluonFunction
from mxfusion.inference import VariationalPosteriorForwardSampling
from pylab import *
import GPy
import matplotlib
import numpy as np
import horovod.mxnet as hvd


np.random.seed(0)
k = GPy.kern.RBF(1, lengthscale=0.1)
x = np.random.rand(1000,1)
y = np.random.multivariate_normal(mean=np.zeros((1000,)), cov=k.K(x), size=(1,)).T
matplotlib.pyplot.plot(x[:,0], y[:,0], '.')

import mxnet as mx

hvd.init()

D = 50
net = nn.HybridSequential(prefix='nn_')
with net.name_scope():
    net.add(nn.Dense(D, activation="tanh", in_units=1))
    net.add(nn.Dense(D, activation="tanh", in_units=D))
    net.add(nn.Dense(1, flatten=True, in_units=D))
net.initialize(mx.init.Xavier(magnitude=3))

m = Model()
m.N = Variable()
m.f = MXFusionGluonFunction(net, num_outputs=1,broadcastable=False)
m.x = Variable(shape=(m.N,1))
m.v = Variable(shape=(1,), transformation=PositiveTransformation(), initial_value=mx.nd.array([0.01]))
m.r = m.f(m.x)
from mxfusion.components.functions.operators import broadcast_to

for v in m.r.factor.parameters.values():
    v.set_prior(Normal(mean=broadcast_to(mx.nd.array([0], dtype='float64'), v.shape),
                       variance=broadcast_to(mx.nd.array([1.], dtype='float64'), v.shape)))
m.y = Normal.define_variable(mean=m.r, variance=broadcast_to(m.v, (m.N,1)), shape=(m.N,1))

from mxfusion.inference import BatchInferenceLoop, create_Gaussian_meanfield, DistributedGradBasedInference, StochasticVariationalInference

dtype = 'float64'

observed = [m.y, m.x]
q = create_Gaussian_meanfield(model=m, observed=observed)


alg = StochasticVariationalInference(num_samples=3, model=m, posterior=q, observed=observed)
infr = DistributedGradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())

infr.initialize(y=(1000,1), x=(1000,1))
for v_name, v in m.r.factor.parameters.items():
    infr.params[q[v].factor.mean] = net.collect_params()[v_name].data()
    infr.params[q[v].factor.variance] = mx.nd.ones_like(infr.params[q[v].factor.variance])*1e-6

infr.run(max_iter=2000, learning_rate=1e-2, y=mx.nd.array(y, dtype=dtype), x=mx.nd.array(x, dtype=dtype), verbose=True)

xt = np.linspace(0,1,100)[:,None]

infr2 = VariationalPosteriorForwardSampling(10, [m.x], infr, [m.r])
res = infr2.run(x=mx.nd.array(xt, dtype=dtype))
yt = res[0].asnumpy()
yt_mean = yt.mean(0)
yt_std = yt.std(0)


for i in range(yt.shape[0]):
    matplotlib.pyplot.plot(xt[:,0],yt[i,:,0],'k',alpha=0.2)
matplotlib.pyplot.plot(x[:,0],y[:,0],'.')

matplotlib.pyplot.show()
