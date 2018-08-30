from ..variables import Variable
from .univariate import UnivariateDistribution, UnivariateLogPDFDecorator, UnivariateDrawSamplesDecorator
from ...util.customop import broadcast_to_w_samples


class PointMass(UnivariateDistribution):
    """
    The Point Mass distribution.

    :param value: the location of the point mass.
    """
    def __init__(self, location, rand_gen=None, dtype=None, ctx=None):
        inputs = [('location', location)]
        input_names = ['location']
        output_names = ['random_variable']
        super(PointMass, self).__init__(inputs=inputs, outputs=None,
                                        input_names=input_names,
                                        output_names=output_names,
                                        rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    @UnivariateLogPDFDecorator()
    def log_pdf(self, location, random_variable, F=None):
        """
        Computes the logaorithm of probabilistic density function of the normal distribution.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param location: the location of the point mass.
        :param random_variable: the point to compute the logpdf for.
        :returns: An operator chain to compute the logpdf of the Normal distribution.
        """
        return 0.

    @UnivariateDrawSamplesDecorator()
    def draw_samples(self, location, rv_shape, num_samples=1, F=None):
        return broadcast_to_w_samples(F, location, False, (num_samples,) +
                                      rv_shape)

    @staticmethod
    def define_variable(location, shape=None, rand_gen=None, dtype=None,
                        ctx=None):
        """
        Creates and returns a random variable drawn from a Normal distribution.

        :param location: the location of the point mass.
        :param shape: Shape of random variables drawn from the distribution. If non-scalar, each variable is drawn iid.

        :returns: RandomVariable drawn from the distribution specified.
        """
        if not isinstance(location, Variable):
            loc = Variable(value=location)

        p = PointMass(location=loc, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        p._generate_outputs(shape=shape)
        return p.random_variable
