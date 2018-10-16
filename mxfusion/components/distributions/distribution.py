from ..factor import Factor
from .random_gen import MXNetRandomGenerator
from ...util.inference import realize_shape
from ...common.config import get_default_dtype


class LogPDFDecorator(object):
    """
    The decorator for the log_pdf function in Distribution
    """
    def __call__(self, func):

        func_reshaped = self._wrap_log_pdf_with_broadcast(func)
        func_variables = self._wrap_log_pdf_with_variables(func_reshaped)
        return func_variables

    def _wrap_log_pdf_with_variables(self, func):

        def log_pdf_variables(self, F, variables, targets=None):
            """
            Computes the logrithm of the probability density/mass function
            (PDF/PMF) of the distribution. The inputs and outputs variables are
            fetched from the *variables* argument according to their UUIDs.

            :param F: the MXNet computation mode
            :type F: mxnet.symbol or mxnet.ndarray
            :param variables: the set of MXNet arrays that holds the values of
            variables at runtime.
            :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
            :returns: log pdf of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol
            """
            args = {}
            for name, var in self.inputs:
                args[name] = variables[var.uuid]
            for name, var in self.outputs:
                args[name] = variables[var.uuid]
            return func(self, F=F, **args)
        return log_pdf_variables

    def _wrap_log_pdf_with_broadcast(self, func):
        raise NotImplementedError


class DrawSamplesDecorator(object):
    """
    The decorator for the draw_samples function in Distribution
    """
    def __call__(self, func):

        func_reshaped = self._wrap_draw_samples_with_broadcast(func)
        func_variables = self._wrap_draw_samples_with_variables(func_reshaped)
        return func_variables

    def _wrap_draw_samples_with_variables(self, func):

        def draw_samples_variables(self, F, variables, num_samples=1,
                                   always_return_tuple=False, targets=None):
            """
            Draw a set of samples from the distribution. The inputs variables
            are fetched from the *variables* argument according to their UUIDs.

            :param F: the MXNet computation mode
            :type F: mxnet.symbol or mxnet.ndarray
            :param variables: the set of MXNet arrays that holds the values of
            variables at runtime.
            :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
            :param num_samples: the number of drawn samples (default: one)
            :int num_samples: int
            :param always_return_tuple: Whether return a tuple even if there is
            only one variables in outputs.
            :type always_return_tuple: boolean
            :returns: a set samples of the distribution
            :rtypes: MXNet NDArray or MXNet Symbol or [MXNet NDArray or MXNet
            Symbol]
            """
            args = {}
            for name, var in self.inputs:
                args[name] = variables[var.uuid]
            args['rv_shape'] = {name: realize_shape(rv.shape, variables) for
                                name, rv in self.outputs}
            return func(self, F=F, num_samples=num_samples,
                        always_return_tuple=always_return_tuple, **args)
        return draw_samples_variables

    def _wrap_draw_samples_with_broadcast(self, func):
        raise NotImplementedError


class Distribution(Factor):
    """
    The base class of a probability distribution associated with one or a set of random variables.

    :param inputs: the input variables that parameterize the probability distribution.
    :type inputs: {name: Variable}
    :param outputs: the random variables drawn from the distribution.
    :type outputs: {name: Variable}
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, inputs, outputs, input_names, output_names, rand_gen=None, dtype=None, ctx=None):
        super(Distribution, self).__init__(inputs=inputs, outputs=outputs,
                                           input_names=input_names,
                                           output_names=output_names)
        self._rand_gen = MXNetRandomGenerator if rand_gen is None else \
            rand_gen
        self.dtype = get_default_dtype() if dtype is None else dtype
        self.ctx = ctx
        self.log_pdf_scaling = 1

    def replicate_self(self, attribute_map=None):
        replicant = super(Distribution, self).replicate_self(attribute_map)
        replicant._rand_gen = self._rand_gen
        replicant.dtype = self.dtype
        replicant.ctx = self.ctx
        replicant.log_pdf_scaling = 1
        return replicant

    def log_pdf(self, F=None, **kwargs):
        """
        Computes the logarithm of the probability density/mass function (PDF/PMF) of the distribution.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        raise NotImplementedError

    def log_cdf(self, F=None, **kwargs):
        """
        Computes the logarithm of the cumulative distribution function (CDF) of the distribution.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log cdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        raise NotImplementedError

    def draw_samples(self, rv_shape, num_samples=1, F=None, **kwargs):
        """
        Draw a number of samples from the distribution.

        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple, [tuple]
        :param num_samples: the number of drawn samples (default: one).
        :int num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol or [MXNet NDArray or MXNet Symbol]
        """
        raise NotImplementedError

    @staticmethod
    def define_variable(shape=None, rand_gen=None, dtype=None, ctx=None, **kwargs):
        """
        Define a random variable that follows from the specified distribution.

        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :param kwargs: the input variables that parameterize the probability
        distribution.
        :type kwargs: {name: Variable}
        :returns: the random variables drawn from the distribution.
        :rtypes: Variable or [Variable]
        """
        raise NotImplementedError
