from ...common.config import get_default_MXNet_mode
from ..variables import Variable
from .univariate import UnivariateDistribution, UnivariateLogPDFDecorator, UnivariateDrawSamplesDecorator


class Uniform(UnivariateDistribution):
    """
    The one-dimensional Uniform distribution over the half-open interval [low, high) (includes low, but excludes high).
    The Uniform distribution can be defined over a scalar random variable
    or an array of random variables. In case of an array of random variables, the low and high are broadcasted
    to the shape of the output random variable (array).

    :param low: Low boundary of the Uniform distribution.
    :type low: Variable
    :param high: High boundary of the Uniform distribution.
    :type high: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, low, high, rand_gen=None, dtype=None, ctx=None):
        if not isinstance(low, Variable):
            low = Variable(value=low)
        if not isinstance(high, Variable):
            high = Variable(value=high)

        inputs = [('low', low), ('high', high)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(Uniform, self).__init__(inputs=inputs, outputs=None,
                                      input_names=input_names,
                                      output_names=output_names,
                                      rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    @UnivariateLogPDFDecorator()
    def log_pdf(self, low, high, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (PDF) of the Uniform distribution.

        :param low: the low boundary of the Uniform distribution.
        :type low: MXNet NDArray or MXNet Symbol
        :param high: the high boundary of the Uniform distributions.
        :type high: MXNet NDArray or MXNet Symbol
        :param random_variable: the random variable of the Uniform distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        # next 3 lines are the broadcasting equivalent of clip(random_variable, low, high)
        lower_safe = (low - random_variable) <= 0
        upper_safe = (high - random_variable) > 0
        in_bounds = F.broadcast_mul(lower_safe, upper_safe)
        log_likelihood = F.where(
            in_bounds,
            -F.log(F.broadcast_minus(high, low)),
            F.log(F.zeros_like(random_variable))) * self.log_pdf_scaling
        return log_likelihood

    @UnivariateDrawSamplesDecorator()
    def draw_samples(self, low, high, rv_shape, num_samples=1, F=None):
        """
        Draw samples from the Uniform distribution over the half-open interval [low, high)
        (includes low, but excludes high).

        :param low: the low boundary of the Uniform distribution.
        :type low: MXNet NDArray or MXNet Symbol
        :param high: the high boundary of the Uniform distributions.
        :type high: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :int num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the Uniform distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        out_shape = (num_samples,) + rv_shape
        return self._rand_gen.sample_uniform(low=low, high=high,
                                             shape=out_shape, dtype=self.dtype, ctx=self.ctx, F=F)

    @staticmethod
    def define_variable(low=0, high=1, shape=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a Uniform distribution.

        :param low: Low boundary of the distribution.
        :param high: High boundary of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the Uniform distribution.
        :rtypes: Variable
        """
        var = Uniform(low=low, high=high, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        var._generate_outputs(shape=shape)
        return var.random_variable
