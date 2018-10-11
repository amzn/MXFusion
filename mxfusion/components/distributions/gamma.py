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
    Gamma distribution.
    Takes dependency on Scipy to compute the log-gamma function.

    :param a: the alpha parameter of the Gamma distribution.
    :type a: Variable
    :param a: beta parameter of the Gamma distribution.
    :type b: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, a, b, mean_variance_parameterized=False, rand_gen=None, dtype=None, ctx=None):
        if not isinstance(a, Variable):
            a = Variable(value=a)
        if not isinstance(b, Variable):
            b = Variable(value=b)

        self.mean_variance_parameterized = mean_variance_parameterized
        inputs = [('a', a), ('b', b)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(Gamma, self).__init__(inputs=inputs, outputs=None,
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
        if self.mean_variance_parameterized:
            beta = a / b
            alpha = a * beta
            return alpha, beta
        else:
            return a,b

    @UnivariateLogPDFDecorator()
    def log_pdf(self, a, b, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (PDF) of the Gamma distribution.

        :param random_variable: the random variable of the Gamma distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        from ..functions.operators.mxnet_custom_operators import gammaln

        alpha, beta = self._get_alpha_beta(a,b)
        g_alpha = gammaln(alpha)
        p1 = (alpha - 1.) * F.log(random_variable)
        return (p1 - beta * random_variable) - (g_alpha - alpha * F.log(beta))

    @UnivariateDrawSamplesDecorator()
    def draw_samples(self, a, b, rv_shape, num_samples=1, F=None):
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
        alpha, beta = self._get_alpha_beta(a,b)
        return F.random.gamma(alpha=alpha, beta=beta, dtype=self.dtype, ctx=self.ctx)

    @staticmethod
    def define_variable(a=0., b=1., shape=None, rand_gen=None,
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
        dist = Gamma(a=a, b=b, rand_gen=rand_gen,
                        dtype=dtype, ctx=ctx)
        dist._generate_outputs(shape=shape)
        return dist.random_variable

    @staticmethod
    def define_variable_mv(mean=0., variance=1., shape=None, rand_gen=None,
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
        dist = Gamma(a=mean, b=variance, mean_variance_parameterized=True, rand_gen=rand_gen,
                        dtype=dtype, ctx=ctx)
        dist._generate_outputs(shape=shape)
        return dist.random_variable
