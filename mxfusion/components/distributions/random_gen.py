# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================


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
    def sample_multinomial(data, get_prob=True, dtype='int32', F=None):
        pass

    @staticmethod
    def sample_bernoulli(prob_true=0.5, dtype='bool', F=None):
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

        if F is mx.ndarray:
            # This is required because MXNet uses _Null instead of None as shape default
            if shape is None:
                return func(dtype=dtype, ctx=ctx, out=out, **kwargs)
            else:
                return func(shape=shape, dtype=dtype, ctx=ctx, out=out, **kwargs)
        else:
            return func(shape=shape, dtype=dtype, out=out, **kwargs)

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
    def sample_bernoulli(prob_true=0.5, dtype=None, shape=None, F=None):
        """
        Sample Bernoulli distributed variables

        :param shape: Array shape of samples
        :param prob_true: Probability of being true
        :param dtype: data type
        :param F: MXNet node
        :return: Array of samples
        """
        F = get_default_MXNet_mode() if F is None else F
        return F.random.uniform(low=0, high=1, shape=shape, dtype=dtype) > prob_true

    @staticmethod
    def sample_gamma(alpha=1, beta=1, shape=None, dtype=None, out=None, ctx=None, F=None):
        """
        Sample Gamma distributed variables

        :param alpha: Also known as shape
        :param beta: Also known as rate
        :param shape: The number of samples to draw. If shape is, e.g., (m, n) and alpha and beta are scalars, output
            shape will be (m, n). If alpha and beta are NDArrays with shape, e.g., (x, y), then output will have shape
            (x, y, m, n), where m*n samples are drawn for each [alpha, beta) pair.
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
