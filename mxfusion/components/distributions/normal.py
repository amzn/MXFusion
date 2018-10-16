import numpy as np
import mxnet as mx
from ...common.config import get_default_MXNet_mode
from ...common.exceptions import InferenceError
from ..variables import Variable
from .univariate import UnivariateDistribution, UnivariateLogPDFDecorator, UnivariateDrawSamplesDecorator
from .distribution import Distribution, LogPDFDecorator, DrawSamplesDecorator
from ..variables import is_sampled_array, get_num_samples
from ...util.customop import broadcast_to_w_samples


class Normal(UnivariateDistribution):
    """
    The one-dimensional normal distribution. The normal distribution can be defined over a scalar random variable or an array of random variables. In case
    of an array of random variables, the mean and variance are broadcasted to the shape of the output random variable (array).

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
    def __init__(self, mean, variance, rand_gen=None, dtype=None, ctx=None):
        if not isinstance(mean, Variable):
            mean = Variable(value=mean)
        if not isinstance(variance, Variable):
            variance = Variable(value=variance)

        inputs = [('mean', mean), ('variance', variance)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(Normal, self).__init__(inputs=inputs, outputs=None,
                                     input_names=input_names,
                                     output_names=output_names,
                                     rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    @UnivariateLogPDFDecorator()
    def log_pdf(self, mean, variance, random_variable, F=None):
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

    @UnivariateDrawSamplesDecorator()
    def draw_samples(self, mean, variance, rv_shape, num_samples=1, F=None):
        """
        Draw samples from the normal distribution.

        :param mean: the mean of the normal distribution.
        :type mean: MXNet NDArray or MXNet Symbol
        :param variance: the variance of the normal distributions.
        :type variance: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :int num_samples: int
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


class MultivariateNormalLogPDFDecorator(LogPDFDecorator):

    def _wrap_log_pdf_with_broadcast(self, func):
        def log_pdf_broadcast(self, F, **kw):
            """
            Computes the logarithm of the probability density/mass function (PDF/PMF) of the distribution. The inputs and outputs variables are in RTVariable format.

            Shape assumptions:
            * mean is S x N x D
            * covariance is S x N x D x D
            * random_variable is S x N x D

            Where:
            * S, the number of samples, is optional. If more than one of the variables has samples, the number of samples in each variable must be the same. S is 1 by default if not a sampled variable.
            * N is the number of data points. N can be any number of dimensions (N_1, N_2, ...) but must be broadcastable to the shape of random_variable.
            * D is the dimension of the distribution.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param kw: the dict of input and output variables of the distribution
            :type kw: {str (name): MXNet NDArray or MXNet Symbol}
            :returns: log pdf of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol
            """
            variables = {name: kw[name] for name, _ in self.inputs}
            variables['random_variable'] = kw['random_variable']
            rv_shape = variables['random_variable'].shape[1:]

            nSamples = max([get_num_samples(F, v) for v in variables.values()])

            shapes_map = {}
            shapes_map['mean'] = (nSamples,) + rv_shape
            shapes_map['covariance'] = (nSamples,) + rv_shape + (rv_shape[-1],)
            shapes_map['random_variable'] = (nSamples,) + rv_shape
            variables = {name: broadcast_to_w_samples(F, v, shapes_map[name])
                         for name, v in variables.items()}
            res = func(self, F=F, **variables)
            return res
        return log_pdf_broadcast


class MultivariateNormalDrawSamplesDecorator(DrawSamplesDecorator):

    def _wrap_draw_samples_with_broadcast(self, func):
        def draw_samples_broadcast(self, F, rv_shape, num_samples=1,
                                   always_return_tuple=False, **kw):
            """
            Draw a number of samples from the distribution. The inputs and outputs variables are in RTVariable format.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param rv_shape: the shape of each sample
            :type rv_shape: tuple
            :param num_samples: the number of drawn samples (default: one)
            :int num_samples: int
            :param always_return_tuple: Whether return a tuple even if there is only one variables in outputs.
            :type always_return_tuple: boolean
            :param kw: the dict of input variables of the distribution
            :type kw: {name: MXNet NDArray or MXNet Symbol}
            :returns: a set samples of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol or [MXNet NDArray or MXNet Symbol]
            """
            rv_shape = list(rv_shape.values())[0]
            variables = {name: kw[name] for name, _ in self.inputs}

            isSamples = any([is_sampled_array(F, v) for v in variables.values()])
            if isSamples:
                num_samples_inferred = max([get_num_samples(F, v) for v in
                                           variables.values()])
                if num_samples_inferred != num_samples:
                    raise InferenceError("The number of samples in the nSamples argument of draw_samples of Normal distribution must be the same as the number of samples given to the inputs. nSamples: "+str(num_samples)+" the inferred number of samples from inputs: "+str(num_samples_inferred)+".")

            shapes_map = {}
            shapes_map['mean'] = (num_samples,) + rv_shape
            shapes_map['covariance'] = (num_samples,) + rv_shape + (rv_shape[-1],)
            shapes_map['random_variable'] = (num_samples,) + rv_shape
            variables = {name: broadcast_to_w_samples(F, v, shapes_map[name])
                         for name, v in variables.items()}

            res = func(self, F=F, rv_shape=rv_shape, num_samples=num_samples,
                       **variables)
            if always_return_tuple:
                res = (res,)
            return res
        return draw_samples_broadcast


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
        self.minibatch_ratio = minibatch_ratio
        if not isinstance(mean, Variable):
            mean = Variable(value=mean)
        if not isinstance(covariance, Variable):
            covariance = Variable(value=covariance)

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
        replicant.minibatch_ratio = self.minibatch_ratio
        return replicant

    @MultivariateNormalLogPDFDecorator()
    def log_pdf(self, mean, covariance, random_variable, F=None):
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
        return 0.5 * (sqnorm_z - (N * np.log(2 * np.pi))) + logdetl

    @MultivariateNormalDrawSamplesDecorator()
    def draw_samples(self, mean, covariance, rv_shape, num_samples=1, F=None):
        """
        Draw a number of samples from the normal distribution.

        :param mean: the mean of the normal distribution.
        :type mean: MXNet NDArray or MXNet Symbol
        :param covariance: the covariance of the normal distributions.
        :type covariance: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :int num_samples: int
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
    def define_variable(shape, mean=0., covariance=None, rand_gen=None,
                        minibatch_ratio=1., dtype=None, ctx=None):
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
