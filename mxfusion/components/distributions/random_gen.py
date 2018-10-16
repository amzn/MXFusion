from abc import ABC
import mxnet as mx
from ...common.config import get_default_dtype, get_default_MXNet_mode


class RandomGenerator(ABC):
    """
    The abstract class of the pseudo-random number generator.
    """

    @staticmethod
    def sample_normal(loc=0, scale=1, shape=None, dtype=None, out=None,
                      ctx=None):
        pass


class MXNetRandomGenerator(RandomGenerator):
    """
    The MXNet pseudo-random number generator.
    """

    @staticmethod
    def sample_normal(loc=0, scale=1, shape=None, dtype=None, out=None,
                      ctx=None, F=None):
        """
        :param ref_var: reference variable for getting the context of execution.
        """
        dtype = get_default_dtype() if dtype is None else dtype
        F = get_default_MXNet_mode() if F is None else F
        if F is mx.ndarray:
            return F.random.normal(loc=loc, scale=scale, shape=shape,
                                   dtype=dtype, ctx=ctx, out=out)
        else:
            return F.random.normal(loc=loc, scale=scale, shape=shape,
                                   dtype=dtype, out=out)

    @staticmethod
    def sample_multinomial(data, get_prob=True, dtype='int32',
                           F=None):
        F = get_default_MXNet_mode() if F is None else F
        return F.random.multinomial(
            data=data, get_prob=get_prob, dtype=dtype)

    @staticmethod
    def sample_gamma(alpha=1, beta=1, shape=None, dtype=None, out=None,
                     ctx=None, F=None):
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
        return MXNetRandomGenerator.sample_univariate(
            func=F.random.gamma, alpha=alpha, beta=beta,
            shape=shape, dtype=dtype, out=out, ctx=ctx, F=F)
