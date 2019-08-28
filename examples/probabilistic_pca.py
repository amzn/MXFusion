import warnings
warnings.filterwarnings('ignore')
import mxfusion as mf
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from mxfusion_gpu import MXFusionGPU


def log_spiral(a,b,t):
    x = a * np.exp(b*t) * np.cos(t)
    y = a * np.exp(b*t) * np.sin(t)
    return np.vstack([x,y]).T

N = 100
D = 100
K = 2

a = 1
b = 0.1
t = np.linspace(0,6*np.pi,N)
r = log_spiral(a,b,t)


mxgpu = MXFusionGPU(4)

plt.plot(r[:,0], r[:,1],'.')
# plt.show()

w = np.random.randn(K,N)
x_train = np.dot(r,w) + np.random.randn(N,N) * 1e-3


dim1 = 71
dim2 = 11
plt.scatter(x_train[:,dim1], x_train[:,dim2], color='blue', alpha=0.1)
plt.axis([-10, 10, -10, 10])
plt.title("Simulated data set")
# plt.show()

from mxfusion.models import Model
import mxnet.gluon.nn as nn
from mxfusion.components import Variable
from mxfusion.components.variables import PositiveTransformation
from mxfusion.components.functions.operators import broadcast_to

m = Model()
m.w = Variable(shape=(K,D), initial_value=mx.nd.array(np.random.randn(K,D)))

dot = nn.HybridLambda(function='dot')
m.dot = mf.functions.MXFusionGluonFunction(dot, num_outputs=1, broadcastable=False)

cov = mx.nd.broadcast_to(mx.nd.expand_dims(mx.nd.array(np.eye(K,K)), 0),shape=(N,K,K))
m.z = mf.distributions.MultivariateNormal.define_variable(mean=mx.nd.zeros(shape=(N,K)), covariance=cov, shape=(N,K))
m.sigma_2 = Variable(shape=(1,), transformation=PositiveTransformation())
m.x = mf.distributions.Normal.define_variable(mean=m.dot(m.z, m.w), variance=broadcast_to(m.sigma_2, (N,D)), shape=(N,D))

from mxfusion.inference import BatchInferenceLoop, GradBasedInference, StochasticVariationalInference
class SymmetricMatrix(mx.gluon.HybridBlock):
    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.sum((F.expand_dims(x, 3)*F.expand_dims(x, 2)), axis=-3)

q = mf.models.Posterior(m)
sym = mf.components.functions.MXFusionGluonFunction(SymmetricMatrix(), num_outputs=1, broadcastable=False)
cov = Variable(shape=(N,K,K), initial_value=mx.nd.broadcast_to(mx.nd.expand_dims(mx.nd.array(np.eye(K,K) * 1e-2), 0),shape=(N,K,K)))
q.post_cov = sym(cov)
q.post_mean = Variable(shape=(N,K), initial_value=mx.nd.array(np.random.randn(N,K)))
q.z.set_prior(mf.distributions.MultivariateNormal(mean=q.post_mean, covariance=q.post_cov))

observed = [m.x]
alg = StochasticVariationalInference(num_samples=3, model=m, posterior=q, observed=observed)
infr = GradBasedInference(inference_algorithm=alg,  grad_loop=BatchInferenceLoop())

infr.initialize(x=mx.nd.array(x_train))
infr.run(max_iter=1000, learning_rate=1e-2, x=mx.nd.array(x_train), multi_processor=mxgpu)

post_z_mean = infr.params[q.z.factor.mean].asnumpy()
plt.plot(post_z_mean[:,0], post_z_mean[:,1],'.')

plt.show()