from ...common.config import get_default_MXNet_mode
from ..variables import Variable
from .univariate import UnivariateDistribution, UnivariateLogPDFDecorator, UnivariateDrawSamplesDecorator


class Laplace(UnivariateDistribution):
    """
    The one-dimensional Laplace distribution. The Laplace distribution can be defined over a scalar random variable
    or an array of random variables. In case of an array of random variables, the location and scale are broadcasted
    to the shape of the output random variable (array).

    :param location: Location of the Laplace distribution.
    :type location: Variable
    :param scale: Scale of the Laplace distribution.
    :type scale: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, location, scale, rand_gen=None, dtype=None, ctx=None):
        if not isinstance(location, Variable):
            location = Variable(value=location)
        if not isinstance(scale, Variable):
            scale = Variable(value=scale)

        inputs = [('location', location), ('scale', scale)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(Laplace, self).__init__(inputs=inputs, outputs=None,
                                      input_names=input_names,
                                      output_names=output_names,
                                      rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    @UnivariateLogPDFDecorator()
    def log_pdf(self, location, scale, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (PDF) of the Laplace distribution.

        :param location: the location of the Laplace distribution.
        :type location: MXNet NDArray or MXNet Symbol
        :param scale: the scale of the Laplace distributions.
        :type scale: MXNet NDArray or MXNet Symbol
        :param random_variable: the random variable of the Laplace distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        logvar = -F.log(2 * scale)
        logL = F.broadcast_minus(logvar, F.broadcast_div(
            F.abs(F.broadcast_minus(random_variable, location)), scale)) * self.log_pdf_scaling
        return logL

    @UnivariateDrawSamplesDecorator()
    def draw_samples(self, location, scale, rv_shape, num_samples=1, F=None):
        """
        Draw samples from the Laplace distribution.

        :param location: the location of the Laplace distribution.
        :type location: MXNet NDArray or MXNet Symbol
        :param scale: the scale of the Laplace distributions.
        :type scale: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :int num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the Laplace distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        out_shape = (num_samples,) + rv_shape
        return F.broadcast_add(F.broadcast_mul(self._rand_gen.sample_laplace(
            shape=out_shape, dtype=self.dtype, ctx=self.ctx),
            scale), location)

    @staticmethod
    def define_variable(location=0., scale=1., shape=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a Laplace distribution.

        :param location: Location of the distribution.
        :param scale: Scale of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the Laplace distribution.
        :rtypes: Variable
        """
        var = Laplace(location=location, scale=scale, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        var._generate_outputs(shape=shape)
        return var.random_variable
