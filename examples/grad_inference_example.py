import warnings
warnings.filterwarnings('ignore')
import numpy as np
from mxfusion import Variable, Model
from mxfusion.components.variables import PositiveTransformation
from mxfusion.components.distributions import Normal

from mxfusion.inference import GradBasedInference, MAP
import mxnet as mx


from mxfusion.common import config
config.DEFAULT_DTYPE = 'float64'

import horovod.mxnet as hvd

hvd.init()

np.random.seed(0)
mean_groundtruth = 3.
variance_groundtruth = 5.
N = 100

data = np.random.randn(N)*np.sqrt(variance_groundtruth) + mean_groundtruth


# m = Model()
# m.mu = Variable()
# m.s = Variable(transformation=PositiveTransformation())
# m.Y = Normal.define_variable(mean=m.mu, variance=m.s, shape=(N,))
# infr = GradBasedInference(inference_algorithm=MAP(model=m, observed=[m.Y]))
# infr.run(Y=mx.nd.array(data, dtype='float64'), learning_rate=0.1, max_iter=2000, verbose=True, multi_processor=True)
# mean_estimated = infr.params[m.mu].asnumpy()
# variance_estimated = infr.params[m.s].asnumpy()
# print('The estimated mean and variance: %f, %f.' % (mean_estimated, variance_estimated))
# print('The true mean and variance: %f, %f.' % (mean_groundtruth, variance_groundtruth))
#
m = Model()

m.mu = Normal.define_variable(mean=mx.nd.array([0], dtype='float64'),
                              variance=mx.nd.array([100], dtype='float64'), shape=(1,))

dtype='float64'

# The mean and standard deviation of the mean parameter is 3.119226(0.222088).
# The 15th, 50th and 85th percentile of the variance parameter is 4.602657, 5.309384 and 6.018709.


from mxfusion.components.functions import MXFusionGluonFunction

m.s_hat = Normal.define_variable(mean=mx.nd.array([5], dtype='float64'),
                                 variance=mx.nd.array([100], dtype='float64'),
                                 shape=(1,), dtype=dtype)
trans_mxnet = mx.gluon.nn.HybridLambda(lambda F, x: F.Activation(x, act_type='softrelu'))
m.trans = MXFusionGluonFunction(trans_mxnet, num_outputs=1, broadcastable=True)
m.s = m.trans(m.s_hat)
m.Y = Normal.define_variable(mean=m.mu, variance=m.s, shape=(N,), dtype=dtype)

from mxfusion.inference import create_Gaussian_meanfield

q = create_Gaussian_meanfield(model=m, observed=[m.Y])

from mxfusion.inference import StochasticVariationalInference

data = np.random.randn(N)*np.sqrt(variance_groundtruth) + mean_groundtruth

infr = GradBasedInference(inference_algorithm=StochasticVariationalInference(
    model=m, posterior=q, num_samples=10, observed=[m.Y]))
infr.run(Y=mx.nd.array(data, dtype='float64'), learning_rate=0.1, verbose=True, multi_processor=True,max_iter=2000)

mu_mean = infr.params[q.mu.factor.mean].asscalar()
mu_std = np.sqrt(infr.params[q.mu.factor.variance].asscalar())
s_hat_mean = infr.params[q.s_hat.factor.mean].asscalar()
s_hat_std = np.sqrt(infr.params[q.s_hat.factor.variance].asscalar())
s_15 = np.log1p(np.exp(s_hat_mean - s_hat_std))
s_50 = np.log1p(np.exp(s_hat_mean))
s_85 = np.log1p(np.exp(s_hat_mean + s_hat_std))
print('The mean and standard deviation of the mean parameter is %f(%f). ' % (mu_mean, mu_std))
print('The 15th, 50th and 85th percentile of the variance parameter is %f, %f and %f.'%(s_15,s_50, s_85))