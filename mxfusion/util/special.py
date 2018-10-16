import numpy as np
from mxfusion.common.config import get_default_MXNet_mode


# noinspection PyPep8Naming
def log_determinant(A, F=None):
    """
    Compute log determinant of the positive semi-definite matrix A. This uses the fact that:

    log|A| = log|LL'| = log|L||L'| = log|L|^2 = 2log|L| = 2 tr(L)
    where L is the Cholesky factor of A

    :param A: Positive semi-definite matrix
    :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
    :return: Log determinant
    """
    F = get_default_MXNet_mode() if F is None else F

    return 2 * F.linalg.sumlogdiag(F.linalg.potrf(A))


# noinspection PyPep8Naming
def log_mv_gamma(x, p, F=None):
    """
    Compute the log of the multivariate gamma function of dimension p
    See https://en.wikipedia.org/wiki/Multivariate_gamma_function

    :param x: input variable
    :param p: dimension
    :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
    :return: log of multivariate gamma function
    """
    F = get_default_MXNet_mode() if F is None else F

    if p == 1:
        # note that \Gamma_1(x) reduces to the ordinary gamma function
        return F.gammaln(x)

    # leading constant
    c = p * (p - 1) / 4 * np.log(np.pi)

    # Sum over univariate gamma functions
    j = F.arange(0, p)
    g = F.gammaln(F.broadcast_sub(x / 2, j / 2))

    return F.sum(g) + c


# noinspection PyPep8Naming
def trace(A, F=None):
    """
    Compute the trace of the matrix A

    :param A: input matrix
    :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
    :return: trace of the matrix
    """
    F = get_default_MXNet_mode() if F is None else F

    return F.sum(F.diag(A))


def solve(A, B, F=None):
    """
    Solve the equation AX = B: X = A^{-1}B
    To compute tr(V{-1} X) we'll first compute the Cholesky decomposition of V:
    A = V^{-1}X
    => VA = X
    LL'A = X
    Then we solve two linear systems involving L:
    Ly = X
    L'A = y

    :param A: input matrix
    :param B: input matrix or vector
    :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
    :return:
    """
    L = F.linalg.potrf(A)
    y = F.linalg.trsm(L, B)
    X = F.linalg.trsm(L, y, transpose=1)
    return X
