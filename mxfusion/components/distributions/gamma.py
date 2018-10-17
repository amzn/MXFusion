import numpy as np
import mxnet as mx
from ...common.config import get_default_MXNet_mode
from ..variables import Variable
from .univariate import UnivariateDistribution, UnivariateLogPDFDecorator, UnivariateDrawSamplesDecorator
from .distribution import Distribution, LogPDFDecorator, DrawSamplesDecorator
from ..variables import is_sampled_array, get_num_samples
from ...util.customop import broadcast_to_w_samples


class Gamma(UnivariateDistribution):
    """
    Gamma distribution parameterized using Alpha and Beta.
    Takes dependency on Scipy to compute the log-gamma function.

    :param alpha: the alpha parameter of the Gamma distribution.
    :type alpha: Variable
    :param beta: beta parameter of the Gamma distribution.
    :type beta: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, alpha, beta, rand_gen=None, dtype=None, ctx=None):
        if not isinstance(alpha, Variable):
            alpha = Variable(value=alpha)
        if not isinstance(beta, Variable):
            beta = Variable(value=beta)

        inputs = [('alpha', alpha), ('beta', beta)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(Gamma, self).__init__(inputs=inputs, outputs=None,
                                     input_names=input_names,
                                     output_names=output_names,
                                     rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    @UnivariateLogPDFDecorator()
    def log_pdf(self, alpha, beta, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (PDF) of the Gamma distribution.

        :param random_variable: the random variable of the Gamma distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        g_alpha = F.gammaln(alpha)
        p1 = (alpha - 1.) * F.log(random_variable)
        return (p1 - beta * random_variable) - (g_alpha - alpha * F.log(beta))

    @UnivariateDrawSamplesDecorator()
    def draw_samples(self, alpha, beta, rv_shape, num_samples=1, F=None):
        """
        Draw samples from the Gamma distribution.
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param nSamples: the number of drawn samples (default: one).
        :int nSamples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the Gamma distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        return F.random.gamma(alpha=alpha, beta=beta, dtype=self.dtype, ctx=self.ctx)

    @staticmethod
    def define_variable(alpha=0., beta=1., shape=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a Gamma distribution parameterized with a and b parameters.

        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the Gamma distribution.
        :rtypes: Variable
        """
        dist = Gamma(alpha=alpha, beta=beta, rand_gen=rand_gen,
                        dtype=dtype, ctx=ctx)
        dist._generate_outputs(shape=shape)
        return dist.random_variable

class GammaMeanVariance(UnivariateDistribution):
    """
    Gamma distribution parameterized using Mean and Variance.
    Takes dependency on Scipy to compute the log-gamma function.

    :param mean: the mean parameter of the Gamma distribution.
    :type mean: Variable
    :param variance: variance parameter of the Gamma distribution.
    :type variance: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, mean, variance, rand_gen=None, dtype=None, ctx=None):
        if not isinstance(mean, Variable):
            mean = Variable(value=mean)
        if not isinstance(variance, Variable):
            variance = Variable(value=variance)

        inputs = [('mean', mean), ('variance', variance)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(GammaMeanVariance, self).__init__(inputs=inputs, outputs=None,
                                     input_names=input_names,
                                     output_names=output_names,
                                     rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    def _get_alpha_beta(self, a, b):
        """
        Returns the alpha/beta representation of the input variables.
        Based on if the variable was parameterized using alpha/beta or with mean/variance.
        :param a: alpha or mean
        :type a: mx.ndarray.array or mx.symbol.array
        :param b: beta or variance
        :type b: mx.ndarray.array or mx.symbol.array
        """
        beta = a / b
        alpha = a * beta
        return alpha, beta

    @UnivariateLogPDFDecorator()
    def log_pdf(self, mean, variance, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (PDF) of the Gamma distribution.

        :param random_variable: the random variable of the Gamma distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        alpha, beta = self._get_alpha_beta(mean, variance)
        g_alpha = F.gammaln(alpha)
        p1 = (alpha - 1.) * F.log(random_variable)
        return (p1 - beta * random_variable) - (g_alpha - alpha * F.log(beta))

    @UnivariateDrawSamplesDecorator()
    def draw_samples(self, mean, variance, rv_shape, num_samples=1, F=None):
        """
        Draw samples from the Gamma distribution.
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param nSamples: the number of drawn samples (default: one).
        :int nSamples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the Gamma distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        alpha, beta = self._get_alpha_beta(mean, variance)
        return F.random.gamma(alpha=alpha, beta=beta, dtype=self.dtype, ctx=self.ctx)

    @staticmethod
    def define_variable(mean=0., variance=1., shape=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a Gamma distribution parameterized with mean and variance.

        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the Gamma distribution.
        :rtypes: Variable
        """
        dist = GammaMeanVariance(mean=mean, variance=variance, rand_gen=rand_gen,
                        dtype=dtype, ctx=ctx)
        dist._generate_outputs(shape=shape)
        return dist.random_variable
