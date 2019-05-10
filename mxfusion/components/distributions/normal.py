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


import numpy as np
import mxnet as mx
import itertools
from ...util.special import log_determinant
from ...common.config import get_default_MXNet_mode
from ..variables import Variable
from .distribution import Distribution
from .univariate import UnivariateDistribution
from ...runtime import distributions


class Normal(UnivariateDistribution):
    """
    The one-dimensional normal distribution. The normal distribution can be defined over a scalar random variable or an
    array of random variables. In case of an array of random variables, the mean and variance are broadcasted to the
    shape of the output random variable (array).

    :param mean: Mean of the normal distribution.
    :type mean: Variable
    :param variance: Variance of the normal distribution.
    :type variance: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    runtime_dist_class = distributions.normal.NormalRunTime

    def __init__(self, mean, variance, rand_gen=None, dtype=None, ctx=None):
        inputs = [('mean', mean), ('variance', variance)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(Normal, self).__init__(inputs=inputs, outputs=None,
                                     input_names=input_names,
                                     output_names=output_names,
                                     rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    def log_pdf_impl(self, mean, variance, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (PDF) of the normal distribution.

        :param mean: the mean of the normal distribution.
        :type mean: MXNet NDArray or MXNet Symbol
        :param variance: the variance of the normal distributions.
        :type variance: MXNet NDArray or MXNet Symbol
        :param random_variable: the random variable of the normal distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        logvar = np.log(2 * np.pi) / -2 + F.log(variance) / -2
        logL = F.broadcast_add(logvar, F.broadcast_div(F.square(
            F.broadcast_minus(random_variable, mean)), -2 * variance)) * self.log_pdf_scaling
        return logL

    def draw_samples_impl(self, mean, variance, rv_shape, num_samples=1, F=None):
        """
        Draw samples from the normal distribution.

        :param mean: the mean of the normal distribution.
        :type mean: MXNet NDArray or MXNet Symbol
        :param variance: the variance of the normal distributions.
        :type variance: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :type num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the normal distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        out_shape = (num_samples,) + rv_shape
        return F.broadcast_add(F.broadcast_mul(self._rand_gen.sample_normal(
            shape=out_shape, dtype=self.dtype, ctx=self.ctx),
            F.sqrt(variance)), mean)

    @staticmethod
    def define_variable(mean=0., variance=1., shape=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a normal distribution.

        :param mean: Mean of the distribution.
        :param variance: Variance of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the normal distribution.
        :rtypes: Variable
        """
        normal = Normal(mean=mean, variance=variance, rand_gen=rand_gen,
                        dtype=dtype, ctx=ctx)
        normal._generate_outputs(shape=shape)
        return normal.random_variable


class MultivariateNormal(Distribution):
    """
    The multi-dimensional normal distribution.

    :param mean: Mean of the normal distribution.
    :type mean: Variable
    :param covariance: Covariance matrix of the distribution.
    :type covariance: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, mean, covariance, rand_gen=None, minibatch_ratio=1.,
                 dtype=None, ctx=None):
        inputs = [('mean', mean), ('covariance', covariance)]
        input_names = ['mean', 'covariance']
        output_names = ['random_variable']
        super(MultivariateNormal, self).__init__(inputs=inputs, outputs=None,
                                     input_names=input_names,
                                     output_names=output_names,
                                     rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    def replicate_self(self, attribute_map=None):
        """
        Replicates this Factor, using new inputs, outputs, and a new uuid.
        Used during model replication to functionally replicate a factor into a new graph.

        :param inputs: new input variables of the factor.
        :type inputs: a dict of {'name' : Variable} or None
        :param outputs: new output variables of the factor.
        :type outputs: a dict of {'name' : Variable} or None
        """
        replicant = super(MultivariateNormal, self).replicate_self(attribute_map)
        return replicant

    def log_pdf_impl(self, mean, covariance, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (PDF) of the normal distribution.

        :param mean: the mean of the normal distribution.
        :type mean: MXNet NDArray or MXNet Symbol
        :param covariance: the covariance of the distribution.
        :type covariance: MXNet NDArray or MXNet Symbol
        :param random_variable: the random variable of the normal distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        N = mean.shape[-1]
        lmat = F.linalg.potrf(covariance)
        logdetl = - F.linalg.sumlogdiag(F.abs(lmat)) # maybe sum if n x d x d
        targets = random_variable - mean
        zvec = F.sum(F.linalg.trsm(lmat, F.expand_dims(targets, axis=-1)), axis=-1)
        sqnorm_z = - F.sum(F.square(zvec), axis=-1)
        return (0.5 * (sqnorm_z - (N * np.log(2 * np.pi))) + logdetl)* self.log_pdf_scaling

    def draw_samples_impl(self, mean, covariance, rv_shape, num_samples=1, F=None):
        """
        Draw a number of samples from the normal distribution.

        :param mean: the mean of the normal distribution.
        :type mean: MXNet NDArray or MXNet Symbol
        :param covariance: the covariance of the normal distributions.
        :type covariance: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :type num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the normal distribution
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        out_shape = (num_samples,) + rv_shape + (1,)
        lmat = F.linalg.potrf(covariance)
        epsilon = self._rand_gen.sample_normal(
            shape=out_shape, dtype=self.dtype, ctx=self.ctx)
        lmat_eps = F.linalg.trmm(lmat, epsilon)
        return F.broadcast_add(lmat_eps.sum(-1), mean)

    @staticmethod
    def define_variable(shape, mean=0., covariance=None, rand_gen=None, minibatch_ratio=1., dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a normal distribution.

        :param mean: Mean of the distribution.
        :param covariance: Variance of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the normal distribution.
        :rtypes: Variable
        """
        covariance = covariance if covariance is not None else mx.nd.array(np.eye(N=shape[-1]), dtype=dtype, ctx=ctx)
        normal = MultivariateNormal(mean=mean, covariance=covariance,
                                    rand_gen=rand_gen,
                                    dtype=dtype, ctx=ctx)
        normal._generate_outputs(shape=shape)
        return normal.random_variable

    def _generate_outputs(self, shape):
        """
        Set the output variable of the distribution.

        :param shape: the shape of the random distribution.
        :type shape: tuple
        """
        self.outputs = [('random_variable', Variable(value=self, shape=shape))]


class NormalMeanPrecision(UnivariateDistribution):
    """
    The one-dimensional normal distribution, parameterized by mean and precision rather than mean and variance.
    The normal distribution can be defined over a scalar random variable
    or an array of random variables. In case of an array of random variables, the mean and precisions are broadcasted
    to the shape of the output random variable (array).

    :param mean: Mean of the normal distribution.
    :type mean: Variable
    :param precision: Precision of the normal distribution.
    :type precision: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, mean, precision, rand_gen=None, dtype=None, ctx=None):
        inputs = [('mean', mean), ('precision', precision)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(NormalMeanPrecision, self).__init__(inputs=inputs, outputs=None,
                                                  input_names=input_names,
                                                  output_names=output_names,
                                                  rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    def log_pdf_impl(self, mean, precision, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (PDF) of the normal distribution.

        :param mean: the mean of the normal distribution.
        :type mean: MXNet NDArray or MXNet Symbol
        :param precision: the precision of the normal distributions.
        :type precision: MXNet NDArray or MXNet Symbol
        :param random_variable: the random variable of the normal distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        logvar = (F.log(precision) - np.log(2 * np.pi)) / 2
        logL = F.broadcast_add(logvar, F.broadcast_mul(F.square(
            F.broadcast_minus(random_variable, mean)), -precision / 2)) * self.log_pdf_scaling
        return logL

    def draw_samples_impl(self, mean, precision, rv_shape, num_samples=1, F=None):
        """
        Draw samples from the normal distribution.

        :param mean: the mean of the normal distribution.
        :type mean: MXNet NDArray or MXNet Symbol
        :param precision: the precision of the normal distributions.
        :type precision: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :type num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the normal distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        out_shape = (num_samples,) + rv_shape
        return F.broadcast_add(F.broadcast_div(self._rand_gen.sample_normal(
            shape=out_shape, dtype=self.dtype, ctx=self.ctx),
            F.sqrt(precision)), mean)

    @staticmethod
    def define_variable(mean=0., precision=1., shape=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a normal distribution.

        :param mean: Mean of the distribution.
        :param precision: Precision of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the normal distribution.
        :rtypes: Variable
        """
        normal = NormalMeanPrecision(mean=mean, precision=precision, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        normal._generate_outputs(shape=shape)
        return normal.random_variable


class MultivariateNormalMeanPrecision(Distribution):
    """
    The multi-dimensional normal distribution parameterized by mean and precision rather than mean and variance.

    :param mean: Mean of the normal distribution.
    :type mean: Variable
    :param precision: Precision matrix of the distribution.
    :type precision: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, mean, precision, rand_gen=None, minibatch_ratio=1.,
                 dtype=None, ctx=None):
        inputs = [('mean', mean), ('precision', precision)]
        input_names = ['mean', 'precision']
        output_names = ['random_variable']
        super(MultivariateNormalMeanPrecision, self).__init__(
            inputs=inputs, outputs=None, input_names=input_names,
            output_names=output_names, rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    def replicate_self(self, attribute_map=None):
        """
        Replicates this Factor, using new inputs, outputs, and a new uuid.
        Used during model replication to functionally replicate a factor into a new graph.

        :param inputs: new input variables of the factor.
        :type inputs: a dict of {'name' : Variable} or None
        :param outputs: new output variables of the factor.
        :type outputs: a dict of {'name' : Variable} or None
        """
        replicant = super(MultivariateNormalMeanPrecision, self).replicate_self(attribute_map)
        return replicant

    def log_pdf_impl(self, mean, precision, random_variable, F=None):
        """
        Computes the logarithm of the probability density function (PDF) of the normal distribution.

        :param mean: the mean of the normal distribution.
        :type mean: MXNet NDArray or MXNet Symbol
        :param precision: the precision of the distribution.
        :type precision: MXNet NDArray or MXNet Symbol
        :param random_variable: the random variable of the normal distribution.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        N = mean.shape[-1]
        c = N * np.log(2 * np.pi)
        logdetl = -log_determinant(precision)
        targets = random_variable - mean

        # TODO: Should be a way to do this without loops
        sqnorm_z = F.zeros(random_variable.shape[:-1], dtype=self.dtype)
        for ix in itertools.product(*map(range, random_variable.shape[:-1])):
            sqnorm_z[ix] = F.dot(F.dot(targets[ix], precision[ix], transpose_a=True), targets[ix])

        return -0.5 * (sqnorm_z + c + logdetl) * self.log_pdf_scaling

    def draw_samples_impl(self, mean, precision, rv_shape, num_samples=1, F=None):
        """
        Draw a number of samples from the normal distribution.

        :param mean: the mean of the normal distribution.
        :type mean: MXNet NDArray or MXNet Symbol
        :param precision: the precision of the normal distributions.
        :type precision: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :type num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the normal distribution
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        out_shape = (num_samples,) + rv_shape + (1,)

        # Use potri instead of potrf:
        # https://mxnet.incubator.apache.org/api/python/symbol/linalg.html#mxnet.symbol.linalg.potri
        lmat = F.linalg.potri(precision)
        epsilon = self._rand_gen.sample_normal(
            shape=out_shape, dtype=self.dtype, ctx=self.ctx)
        lmat_eps = F.linalg.trmm(lmat, epsilon)
        return F.broadcast_add(lmat_eps.sum(-1), mean)

    @staticmethod
    def define_variable(shape, mean=0., precision=None, rand_gen=None,
                        minibatch_ratio=1., dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a normal distribution.

        :param mean: Mean of the distribution.
        :param precision: Precision of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: the random variables drawn from the normal distribution.
        :rtypes: Variable
        """
        precision = precision if precision is not None else mx.nd.array(np.eye(N=shape[-1]), dtype=dtype, ctx=ctx)
        normal = MultivariateNormalMeanPrecision(mean=mean, precision=precision,
                                                 rand_gen=rand_gen,
                                                 dtype=dtype, ctx=ctx)
        normal._generate_outputs(shape=shape)
        return normal.random_variable

    def _generate_outputs(self, shape):
        """
        Set the output variable of the distribution.

        :param shape: the shape of the random distribution.
        :type shape: tuple
        """
        self.outputs = [('random_variable', Variable(value=self, shape=shape))]
