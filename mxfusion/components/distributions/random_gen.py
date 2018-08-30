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
