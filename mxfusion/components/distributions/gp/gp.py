import numpy as np
from ....common.config import get_default_MXNet_mode
from ....common.exceptions import InferenceError
from ...variables import Variable
from ....util.customop import broadcast_to_w_samples
from ..distribution import Distribution, LogPDFDecorator, DrawSamplesDecorator
from ...variables.runtime_variable import get_num_samples


class GaussianProcessLogPDFDecorator(LogPDFDecorator):

    def _wrap_log_pdf_with_broadcast(self, func):
        def log_pdf_broadcast(self, F, **kw):
            """
            Computes the logrithm of the probability density/mass function (PDF/PMF) of the distribution.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param kw: the dict of input and output variables of the distribution
            :type kw: {name: MXNet NDArray or MXNet Symbol}
            :returns: log pdf of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol
            """
            variables = {name: kw[name] for name, _ in self.inputs}
            variables['random_variable'] = kw['random_variable']
            rv_shape = variables['random_variable'].shape[1:]

            num_samples = max([get_num_samples(F, v) for v in variables.values()])
            X_shape = (num_samples,) + rv_shape[:-1] + (self.kernel.input_dim,)
            rv_shape = (num_samples,) + rv_shape

            variables['X'] = broadcast_to_w_samples(F, variables['X'], X_shape)
            variables['random_variable'] = broadcast_to_w_samples(
                F, variables['random_variable'], rv_shape)
            return func(self, F=F, **variables)
        return log_pdf_broadcast


class GaussianProcessDrawSamplesDecorator(DrawSamplesDecorator):

    def _wrap_draw_samples_with_broadcast(self, func):
        def draw_samples_broadcast(self, F, rv_shape, num_samples=1,
                                   always_return_tuple=False, **kw):
            """
            Draw a number of samples from the distribution.

            :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
            :param rv_shape: the shape of each sample
            :type rv_shape: tuple
            :param nSamples: the number of drawn samples (default: one)
            :int nSamples: int
            :param always_return_tuple: Whether return a tuple even if there is only one variables in outputs.
            :type always_return_tuple: boolean
            :param kw: the dict of input variables of the distribution
            :type kw: {name: MXNet NDArray or MXNet Symbol}
            :returns: a set samples of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol or [MXNet NDArray or MXNet Symbol]
            """
            rv_shape = list(rv_shape.values())[0]
            variables = {name: kw[name] for name, _ in self.inputs}

            num_samples_inferred = max([get_num_samples(F, v) for v in
                                        variables.values()])
            if num_samples_inferred > 1 and num_samples_inferred != num_samples:
                raise InferenceError("The number of samples in the num_samples argument of draw_samples of Gaussian process has to be the same as the number of samples given to the inputs. num_samples: "+str(num_samples)+" the inferred number of samples from inputs: "+str(num_samples_inferred)+".")
            X_shape = (num_samples,) + rv_shape[:-1] + (self.kernel.input_dim,)

            variables['X'] = broadcast_to_w_samples(F, variables['X'], X_shape)
            res = func(self, F=F, rv_shape=rv_shape, num_samples=num_samples,
                       **variables)
            if always_return_tuple:
                res = (res,)
            return res
        return draw_samples_broadcast


class GaussianProcess(Distribution):
    """
    The Gaussian process distribution.

    A Gaussian process consists of a kernel function and a mean function (optional). A collection of GP random variables follows a multi-variate
    normal distribution, where the mean is computed from the mean function (zero, if not given) and the covariance matrix is computed from the kernel
    function, both of which are computed given a collection of inputs.

    :param X: the input variables on which the random variables are conditioned.
    :type X: Variable
    :param kernel: the kernel of Gaussian process.
    :type kernel: Kernel
    :param mean_func: the mean function of Gaussian process.
    :type mean_func: N/A
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, X, kernel, mean_func=None, rand_gen=None, dtype=None,
                 ctx=None):
        if not isinstance(X, Variable):
            X = Variable(value=X)
        inputs = [('X', X)] + [(k, v) for k, v in kernel.parameters.items()]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(GaussianProcess, self).__init__(
            inputs=inputs, outputs=None,
            input_names=input_names, output_names=output_names,
            rand_gen=rand_gen, dtype=dtype,
            ctx=ctx)
        self.mean_func = mean_func
        self.kernel = kernel

    @staticmethod
    def define_variable(X, kernel, shape=None, mean_func=None, rand_gen=None, dtype=None, ctx=None):
        """
        Creates and returns a set of random variables drawn from a Gaussian process.

        :param X: the input variables on which the random variables are conditioned.
        :type X: Variable
        :param kernel: the kernel of Gaussian process.
        :type kernel: Kernel
        :param shape: the shape of the random variable(s) (the default shape is the same shape as *X* but the last dimension is changed to one).
        :type shape: tuple or [tuple]
        :param mean_func: the mean function of Gaussian process.
        :type mean_func: N/A
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        """
        gp = GaussianProcess(X=X, kernel=kernel, mean_func=mean_func,
                             rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        gp.outputs = [('random_variable',
                      Variable(value=gp, shape=X.shape[:-1] + (1,) if
                               shape is None else shape))]
        return gp.random_variable

    @GaussianProcessLogPDFDecorator()
    def log_pdf(self, X, random_variable, F=None, **kernel_params):
        """
        Computes the logarithm of the probability density function (PDF) of the Gaussian process.

        :param X: the input variables on which the random variables are conditioned.
        :type X: MXNet NDArray or MXNet Symbol
        :param random_variable: the random_variable of which log-PDF is computed.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray)
        :param **kernel_params: the set of kernel parameters, provided as keyword arguments.
        :type **kernel_params: {str: MXNet NDArray or MXNet Symbol}
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        D = random_variable.shape[-1]
        F = get_default_MXNet_mode() if F is None else F
        K = self.kernel.K(F, X, **kernel_params)
        L = F.linalg.potrf(K)

        if self.mean_func is not None:
            mean = self.mean_func(F, X)
            random_variable = random_variable - mean
        LinvY = F.linalg.trsm(L, random_variable)
        logdet_l = F.linalg.sumlogdiag(F.abs(L))

        return (- logdet_l * D - F.sum(F.sum(F.square(LinvY) + np.log(2. * np.pi), axis=-1), axis=-1) / 2) * self.log_pdf_scaling

    @GaussianProcessDrawSamplesDecorator()
    def draw_samples(self, X, rv_shape, num_samples=1, F=None, **kernel_params):
        """
        Draw a number of samples from the Gaussian process.

        :param X: the input variables on which the random variables are conditioned.
        :type X: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :int num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :param **kernel_params: the set of kernel parameters, provided as keyword arguments.
        :type **kernel_params: {str: MXNet NDArray or MXNet Symbol}
        :returns: a set samples of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        K = self.kernel.K(F, X, **kernel_params)
        L = F.linalg.potrf(K)

        out_shape = (num_samples,) + rv_shape
        L = broadcast_to_w_samples(F, L, out_shape[:-1] + out_shape[-2:-1])

        die = self._rand_gen.sample_normal(
            shape=out_shape, dtype=self.dtype, ctx=self.ctx)
        rv = F.linalg.trmm(L, die)
        if self.mean_func is not None:
            mean = self.mean_func(F, X)
            rv = rv + mean
        return rv

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for a Gaussian process distribution.
        """
        replicant = super(GaussianProcess, self).replicate_self(attribute_map)
        replicant.mean_func = self.mean_func.replicate_self(attribute_map) \
            if self.mean_func is not None else None
        replicant.kernel = self.kernel.replicate_self(attribute_map)
        return replicant
