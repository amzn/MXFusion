from abc import ABC
import mxnet as mx
from ...common.config import get_default_dtype, get_default_MXNet_mode


class RandomGenerator(ABC):
    """
    The abstract class of the pseudo-random number generator.
    """

    @staticmethod
    def sample_normal(loc=0, scale=1, shape=None, dtype=None, out=None, ctx=None):
        pass

    @staticmethod
    def sample_gamma(alpha=1, beta=1, shape=None, dtype=None, out=None, ctx=None):
        pass


    @staticmethod
    def sample_uniform(low=0., high=1., shape=None, dtype=None, out=None, ctx=None, F=None):
        pass

    @staticmethod
    def sample_laplace(location=0., scale=1., shape=None, dtype=None, out=None, ctx=None, F=None):
        pass

class MXNetRandomGenerator(RandomGenerator):
    """
    The MXNet pseudo-random number generator.
    """

    @staticmethod
    def _sample_univariate(func, shape=None, dtype=None, out=None, ctx=None, F=None, **kwargs):
        """
        Wrapper for univariate sampling functions (Normal, Gamma etc.)

        :param func: The function to use for sampling, e.g. F.random.normal
        :param shape: The shape of the samples
        :param dtype: The data type
        :param out: Output variable
        :param ctx: The execution context
        :param F: MXNet node
        :param kwargs: keyword arguments for the sampling function (loc, scale etc)
        :return: Array of samples
        """
        dtype = get_default_dtype() if dtype is None else dtype

        if shape is None:
            return func(dtype=dtype, ctx=ctx, out=out, **kwargs)
        else:
            return func(shape=shape, dtype=dtype, ctx=ctx, out=out, **kwargs)

    @staticmethod
    def sample_normal(loc=0, scale=1, shape=None, dtype=None, out=None, ctx=None, F=None):
        """
        Sample Normal distributed variables

        :param loc: location (mean)
        :param scale: scale (variance)
        :param shape: Array shape of samples
        :param dtype: Data type
        :param out: Output variable
        :param ctx: execution context
        :param F: MXNet node
        :return: Array of samples
        """
        F = get_default_MXNet_mode() if F is None else F
        return MXNetRandomGenerator._sample_univariate(
            func=F.random.normal, loc=loc, scale=scale,
            shape=shape, dtype=dtype, out=out, ctx=ctx, F=F)

    @staticmethod
    def sample_multinomial(data, get_prob=True, dtype='int32', F=None):
        """
        Sample Multinomial distributed variables

        :param data: An *n* dimensional array whose last dimension has length `k`, where
        `k` is the number of possible outcomes of each multinomial distribution.
        For example, data with shape `(m, n, k)` specifies `m*n` multinomial
        distributions each with `k` possible outcomes.
        :param get_prob: If true, a second array containing log likelihood of the drawn
        samples will also be returned.
        This is usually used for reinforcement learning, where you can provide
        reward as head gradient w.r.t. this array to estimate gradient.
        :param dtype: Data type
        :param F: MXNet node
        :return: Array of samples
        """
        F = get_default_MXNet_mode() if F is None else F
        return F.random.multinomial(
            data=data, get_prob=get_prob, dtype=dtype)

    @staticmethod
    def sample_gamma(alpha=1, beta=1, shape=None, dtype=None, out=None, ctx=None, F=None):
        """
        Sample Gamma distributed variables

        :param alpha: Also known as shape
        :param beta: Also known as rate
        :param shape: Array shape of samples
        :param dtype: Data type
        :param out: output variable
        :param ctx: execution context
        :param F: MXNet node
        :return: Array of samples
        """
        F = get_default_MXNet_mode() if F is None else F

        return MXNetRandomGenerator._sample_univariate(
            func=F.random.gamma, alpha=alpha, beta=beta,
            shape=shape, dtype=dtype, out=out, ctx=ctx, F=F)

    @staticmethod
    def sample_uniform(low=0., high=1., shape=None, dtype=None, out=None, ctx=None, F=None):
        """
        Sample uniformly distributed variables
        Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high).

        :param low: lower boundary of output interval
        :param high: upper boundary of output interval
        :param shape: Array shape of samples
        :param dtype: Data type
        :param out: output variable
        :param ctx: execution context
        :param F: MXNet node
        :return: Array of samples
        """
        F = get_default_MXNet_mode() if F is None else F

        samples = MXNetRandomGenerator._sample_univariate(
            func=F.random.uniform, shape=shape, dtype=dtype, out=out, ctx=ctx, F=F)
        # samples = F.broadcast_add(F.broadcast_mul(samples, F.broadcast_sub(high, low)), low)
        samples = samples * (high - low) + low
        return samples

    @staticmethod
    def sample_laplace(location=0., scale=1., shape=None, dtype=None, out=None, ctx=None, F=None):
        """
        Sample Laplace distributed variables

        :param location: Location parameter (=mean)
        :param scale: (>0) Also known as diversity
        :param shape: Array shape of samples
        :param dtype: Data type
        :param out: output variable
        :param ctx: execution context
        :param F: MXNet node
        :return: Array of samples
        """
        F = get_default_MXNet_mode() if F is None else F

        # Given a random variable U drawn from the uniform distribution in the interval (-1/2,1/2], the random variable
        # X =\mu - b\, \sgn(U)\, \ln(1 - 2 | U |)
        # has a Laplace distribution with parameters \mu and b

        U = MXNetRandomGenerator.sample_uniform(low=-0.5, high=0.5, shape=shape, dtype=dtype, out=out, ctx=ctx, F=F)

        if isinstance(scale, F.NDArray):
            b_sgn_U = F.broadcast_mul(scale, F.sign(U))
        else:
            b_sgn_U = scale * F.sign(U)

        ln_1_2_U = F.log(1 - 2 * F.abs(U))

        if isinstance(location, F.NDArray):
            samples = F.broadcast_minus(location, F.broadcast_mul(b_sgn_U, ln_1_2_U))
        else:
            samples = location - F.broadcast_mul(b_sgn_U, ln_1_2_U)

        return samples
