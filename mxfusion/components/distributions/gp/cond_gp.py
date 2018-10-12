import numpy as np
from ....common.config import get_default_MXNet_mode
from ....common.exceptions import InferenceError
from ...variables.variable import Variable
from ....util.customop import broadcast_to_w_samples
from ..distribution import Distribution, LogPDFDecorator, DrawSamplesDecorator
from ...variables.runtime_variable import get_num_samples


class ConditionalGaussianProcessLogPDFDecorator(LogPDFDecorator):

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
            X_cond_shape = (num_samples,) + rv_shape[:-2] + \
                variables['X_cond'].shape[-2:]
            Y_cond_shape = (num_samples,) + rv_shape[:-2] + \
                variables['Y_cond'].shape[-2:]
            rv_shape = (num_samples,) + rv_shape

            variables['X'] = broadcast_to_w_samples(F, variables['X'], X_shape)
            variables['X_cond'] = broadcast_to_w_samples(
                F, variables['X_cond'], X_cond_shape)
            variables['Y_cond'] = broadcast_to_w_samples(
                F, variables['Y_cond'], Y_cond_shape)
            variables['random_variable'] = broadcast_to_w_samples(
                F, variables['random_variable'], rv_shape)
            return func(self, F=F, **variables)
        return log_pdf_broadcast


class ConditionalGaussianProcessDrawSamplesDecorator(DrawSamplesDecorator):

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
            X_cond_shape = (num_samples,) + rv_shape[:-2] + \
                variables['X_cond'].shape[-2:]
            Y_cond_shape = (num_samples,) + rv_shape[:-2] + \
                variables['Y_cond'].shape[-2:]

            variables['X'] = broadcast_to_w_samples(F, variables['X'], X_shape)
            variables['X_cond'] = broadcast_to_w_samples(
                F, variables['X_cond'], X_cond_shape)
            variables['Y_cond'] = broadcast_to_w_samples(
                F, variables['Y_cond'], Y_cond_shape)
            res = func(self, F=F, rv_shape=rv_shape, num_samples=num_samples,
                       **variables)
            if always_return_tuple:
                res = (res,)
            return res
        return draw_samples_broadcast


class ConditionalGaussianProcess(Distribution):
    """
    The Conditional Gaussian process distribution.

    A Gaussian process consists of a kernel function and a mean function (optional). A collection of GP random variables follows a multi-variate
    normal distribution, where the mean is computed from the mean function (zero, if not given) and the covariance matrix is computed from the kernel
    function, both of which are computed given a collection of inputs.

    The conditional Gaussian process is a Gaussian process distribution which is conditioned over a pair of observation X_cond and Y_cond.

    .. math::
       Y \\sim \\mathcal{N}(Y| K_{*c}K_{cc}^{-1}(Y_C - g(X_c)) + g(X), K_{**} - K_{*c}K_{cc}^{-1}K_{*c}^\\top)

    where :math:`g` is the mean function, :math:`K_{**}` is the covariance matrix over :math:`X`, :math:`K_{*c}` is the cross covariance matrix
    between :math:`X` and :math:`X_{c}` and :math:`K_{cc}` is the covariance matrix over :math:`X_{c}`.

    :param X: the input variables on which the random variables are conditioned.
    :type X: Variable
    :param X_cond: the input variables on which the output variables *Y_Cond* are conditioned.
    :type X_cond: Variable
    :param Y_cond: the output variables on which the random variables are conditioned.
    :type Y_cond: Variable
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
    def __init__(self, X, X_cond, Y_cond, kernel, mean_func=None,
                 rand_gen=None, dtype=None, ctx=None):
        if not isinstance(X, Variable):
            X = Variable(value=X)
        inputs = [('X', X), ('X_cond', X_cond), ('Y_cond', Y_cond)] + \
            [(k, v) for k, v in kernel.parameters.items()]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(ConditionalGaussianProcess, self).__init__(
            inputs=inputs, outputs=None, input_names=input_names,
            output_names=output_names, rand_gen=rand_gen, dtype=dtype,
            ctx=ctx)
        self.mean_func = mean_func
        self.kernel = kernel

    @staticmethod
    def define_variable(X, X_cond, Y_cond, kernel, shape=None, mean_func=None,
                        rand_gen=None, minibatch_ratio=1., dtype=None,
                        ctx=None):
        """
        Creates and returns a set of random variable drawn from a Gaussian process.

        :param X: the input variables on which the random variables are conditioned.
        :type X: Variable
        :param X_cond: the input variables on which the output variables *Y_Cond* are conditioned.
        :type X_cond: Variable
        :param Y_cond: the output variables on which the random variables are conditioned.
        :type Y_cond: Variable
        :param kernel: the kernel of Gaussian process.
        :type kernel: Kernel
        :param shape: the shape of the random variable(s) (the default shape is the same shape as *X* but the last dimension is changed to one.)
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
        gp = ConditionalGaussianProcess(
            X=X, X_cond=X_cond, Y_cond=Y_cond, kernel=kernel,
            mean_func=mean_func, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        gp.outputs = [('random_variable',
                      Variable(value=gp, shape=X.shape[:-1] + (1,) if
                               shape is None else shape))]
        return gp.random_variable

    @ConditionalGaussianProcessLogPDFDecorator()
    def log_pdf(self, X, X_cond, Y_cond, random_variable, F=None,
                **kernel_params):
        """
        Computes the logrithm of the probability density function (PDF) of the condtional Gaussian process.

        .. math::
           \\log p(Y| X_c, Y_c, X) = \\log \\mathcal{N}(Y| K_{*c}K_{cc}^{-1}(Y_C - g(X_c)) + g(X), K_{**} - K_{*c}K_{cc}^{-1}K_{*c}^\\top)

        where :math:`g` is the mean function, :math:`K_{**}` is the covariance matrix over :math:`X`, :math:`K_{*c}` is the cross covariance matrix
        between :math:`X` and :math:`X_{c}` and :math:`K_{cc}` is the covariance matrix over :math:`X_{c}`.

        :param X: the input variables on which the random variables are conditioned.
        :type X: MXNet NDArray or MXNet Symbol
        :param X_cond: the input variables on which the output variables *Y_Cond* are conditioned.
        :type X_cond: MXNet NDArray or MXNet Symbol
        :param Y_cond: the output variables on which the random variables are conditioned.
        :type Y_cond: MXNet NDArray or MXNet Symbol
        :param random_variable: the random_variable of which log-PDF is computed.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :param **kernel_params: the set of kernel parameters, provided as keyword arguments.
        :type **kernel_params: {str: MXNet NDArray or MXNet Symbol}
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        D = random_variable.shape[-1]
        F = get_default_MXNet_mode() if F is None else F
        K = self.kernel.K(F, X, **kernel_params)
        Kc = self.kernel.K(F, X_cond, X, **kernel_params)
        Kcc = self.kernel.K(F, X_cond, **kernel_params)
        Lcc = F.linalg.potrf(Kcc)
        LccInvKc = F.linalg.trsm(Lcc, Kc)
        cov = K - F.linalg.syrk(LccInvKc, transpose=True)
        L = F.linalg.potrf(cov)
        if self.mean_func is not None:
            random_variable = random_variable - self.mean_func(F, X)
            Y_cond = Y_cond - self.mean_func(F, X_cond)
        LccInvY = F.linalg.trsm(Lcc, Y_cond)
        rv_mean = F.linalg.gemm2(LccInvKc, LccInvY, True, False)
        LinvY = F.sum(F.linalg.trsm(L, random_variable - rv_mean), axis=-1)
        logdet_l = F.linalg.sumlogdiag(F.abs(L))

        return (- logdet_l * D - F.sum(F.square(LinvY) + np.log(2. * np.pi),
                axis=-1) / 2) * self.log_pdf_scaling

    @ConditionalGaussianProcessDrawSamplesDecorator()
    def draw_samples(self, X, X_cond, Y_cond, rv_shape, num_samples=1, F=None,
                     **kernel_params):
        """
        Draw a number of samples from the condtional Gaussian process.

        :param X: the input variables on which the random variables are conditioned.
        :type X: MXNet NDArray or MXNet Symbol
        :param X_cond: the input variables on which the output variables *Y_Cond* are conditioned.
        :type X_cond: MXNet NDArray or MXNet Symbol
        :param Y_cond: the output variables on which the random variables are conditioned.
        :type Y_cond: MXNet NDArray or MXNet Symbol
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
        Kc = self.kernel.K(F, X_cond, X, **kernel_params)
        Kcc = self.kernel.K(F, X_cond, **kernel_params)
        Lcc = F.linalg.potrf(Kcc)
        LccInvKc = F.linalg.trsm(Lcc, Kc)
        cov = K - F.linalg.syrk(LccInvKc, transpose=True)
        L = F.linalg.potrf(cov)
        if self.mean_func is not None:
            Y_cond = Y_cond - self.mean_func(F, X_cond)
        LccInvY = F.linalg.trsm(Lcc, Y_cond)
        rv_mean = F.linalg.gemm2(LccInvKc, LccInvY, True, False)

        out_shape = (num_samples,) + rv_shape
        L = broadcast_to_w_samples(F, L, out_shape[:-1] + out_shape[-2:-1])

        die = self._rand_gen.sample_normal(
            shape=out_shape, dtype=self.dtype, ctx=self.ctx)
        rv = F.linalg.trmm(L, die) + rv_mean
        if self.mean_func is not None:
            rv = rv + self.mean_func(F, X)
        return rv

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for a conditional Gaussian process distribution.
        """
        replicant = super(ConditionalGaussianProcess,
                          self).replicate_self(attribute_map)
        replicant.mean_func = self.mean_func.replicate_self(attribute_map) \
            if self.mean_func is not None else None
        replicant.kernel = self.kernel.replicate_self(attribute_map)
        return replicant
