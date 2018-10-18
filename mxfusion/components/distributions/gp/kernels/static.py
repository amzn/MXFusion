from .kernel import NativeKernel
from ....variables import Variable
from ....variables import PositiveTransformation
from .....util.customop import broadcast_to_w_samples


class Bias(NativeKernel):
    """
    Bias kernel, which produces a constant value for every entries of the covariance matrix.

    .. math::
       k(x,y) = \\sigma^2

    :param input_dim: the number of dimensions of the kernel. (The total number of active dimensions).
    :type input_dim: int
    :param variance: the initial value for the variance parameter.
    :type variance: float or MXNet NDArray
    :param name: the name of the kernel. The name is used to access kernel parameters.
    :type name: str
    :param active_dims: The dimensions of the inputs that are taken for the covariance matrix computation. (default: None, taking all the dimensions).
    :type active_dims: [int] or None
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    broadcastable = True

    def __init__(self, input_dim, variance=1., name='bias', active_dims=None,
                 dtype=None, ctx=None):
        super(Bias, self).__init__(input_dim=input_dim, name=name,
                                   active_dims=active_dims, dtype=dtype,
                                   ctx=ctx)
        if not isinstance(variance, Variable):
            variance = Variable(shape=(1,),
                                transformation=PositiveTransformation(),
                                initial_value=variance)
        self.variance = variance

    def _compute_K(self, F, X, variance, X2=None):
        """
        The internal interface for the actual covariance matrix computation.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param X2: (optional) the second set of arguments to the kernel. If X2 is None, this computes a square covariance matrix of X. In other words,
            X2 is internally treated as X.
        :type X2: MXNet NDArray or MXNet Symbol
        :param variance: the variance parameter.
        :type variance: MXNet NDArray or MXNet Symbol
        :return: The covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        if X2 is None:
            X2 = X
        return broadcast_to_w_samples(F, variance, X.shape[:-1] +
                                      (X2.shape[-2],))

    def _compute_Kdiag(self, F, X, variance):
        """
        The internal interface for the actual computation for the diagonal.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param variance: the variance parameter.
        :type variance: MXNet NDArray or MXNet Symbol
        :return: The covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        return broadcast_to_w_samples(F, variance, X.shape[:-1])


class White(NativeKernel):
    """
    White kernel, which produces a constant value for the diagonal of the covariance matrix.

    .. math::
       K = \\sigma^2 I

    :param input_dim: the number of dimensions of the kernel. (The total number of active dimensions).
    :type input_dim: int
    :param variance: the initial value for the variance parameter.
    :type variance: float or MXNet NDArray
    :param name: the name of the kernel. The name is used to access kernel parameters.
    :type name: str
    :param active_dims: The dimensions of the inputs that are taken for the covariance matrix computation. (default: None, taking all the dimensions).
    :type active_dims: [int] or None
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    broadcastable = True

    def __init__(self, input_dim, variance=1., name='white', active_dims=None,
                 dtype=None, ctx=None):
        super(White, self).__init__(input_dim=input_dim, name=name,
                                    active_dims=active_dims, dtype=dtype,
                                    ctx=ctx)
        if not isinstance(variance, Variable):
            variance = Variable(shape=(1,),
                                transformation=PositiveTransformation(),
                                initial_value=variance)
        self.variance = variance

    def _compute_K(self, F, X, variance, X2=None):
        """
        The internal interface for the actual covariance matrix computation.

        :param F: MXNet computation type <mx.sym, mx.nd>
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param X2: (optional) the second set of arguments to the kernel. If X2 is None, this computes a square covariance matrix of X. In other words,
            X2 is internally treated as X.
        :type X2: MXNet NDArray or MXNet Symbol
        :param variance: the variance parameter.
        :type variance: MXNet NDArray or MXNet Symbol
        :return: The covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        if X2 is None:
            Imat = F.eye(N=X.shape[-2:-1][0],
                         ctx=self.ctx,
                         dtype=self.dtype)
            Imat = broadcast_to_w_samples(F, Imat, X.shape[:-1] +
                                          X.shape[-2:-1], False)
            return Imat * broadcast_to_w_samples(F, variance, X.shape[:-1] +
                                                 X.shape[-2:-1])
        else:
            return F.zeros(shape=X.shape[:-1] + X2.shape[-2:-1], ctx=self.ctx,
                           dtype=self.dtype)

    def _compute_Kdiag(self, F, X, variance):
        """
        The internal interface for the actual computation for the diagonal of the covariance matrix.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param X: the first set of inputs to the kernel.
        :type X: MXNet NDArray or MXNet Symbol
        :param variance: the variance parameter.
        :type variance: MXNet NDArray or MXNet Symbol
        :return: The covariance matrix.
        :rtype: MXNet NDArray or MXNet Symbol
        """
        return broadcast_to_w_samples(F, variance, X.shape[:-1])
