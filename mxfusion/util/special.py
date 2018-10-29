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

    return 2 * F.linalg.sumlogdiag(F.abs(F.linalg.potrf(A)))


# noinspection PyPep8Naming
def log_multivariate_gamma(x, p, F=None):
    """
    Compute the log of the multivariate gamma function of dimension p
    See https://en.wikipedia.org/wiki/Multivariate_gamma_function
    This will broadcast the first two extra dimensions if necessary

    :param x: input variable
    :param p: dimension
    :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
    :return: log of multivariate gamma function
    """
    F = get_default_MXNet_mode() if F is None else F

    def log_gamma_sum(a):
        # Sum over univariate log gamma functions
        if p == 1:
            # note that \Gamma_1(x) reduces to the ordinary gamma function
            return F.gammaln(a)
        r = F.zeros(p)
        for k in range(1, p + 1):
            r[k - 1] = F.gammaln(a + ((1. - k) / 2))
        return F.sum(r)

    p_mx = F.array([p], dtype=x.dtype)

    # leading constant
    c = p_mx * (p_mx - 1) / 4 * np.log(np.pi)

    shape = x.shape
    x_flat = x.reshape(-1)
    result_flat = F.zeros(x_flat.shape)
    for i in range(len(x_flat)):
        result_flat[i] = log_gamma_sum(x_flat[i])
    result = result_flat.reshape(shape)
    return result + c


# noinspection PyPep8Naming
def trace(A, F=None):
    """
    Compute the trace of the matrix A.
    This will broadcast the first two extra dimensions if necessary

    :param A: input matrix
    :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
    :return: trace of the matrix
    """
    F = get_default_MXNet_mode() if F is None else F

    if A.ndim > 4:
        raise ValueError("Cannot broadcast more than two dimensions")

    if A.ndim == 2:
        F.sum(F.diag(A))

    # TODO: make use of the nd-array support for diag when it's available
    # see: https://github.com/apache/incubator-mxnet/pull/12430

    result = F.zeros(A.shape[:-2])

    if A.ndim == 3:
        for i in range(A.shape[1]):
            result[i] = F.sum(F.diag(A[i, :, :]))

    if A.ndim == 4:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                result[i, j] = F.sum(F.diag(A[i, j, :, :]))

    return result


# noinspection PyPep8Naming
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
    F = get_default_MXNet_mode() if F is None else F

    L = F.linalg.potrf(A)
    y = F.linalg.trsm(L, B)
    X = F.linalg.trsm(L, y, transpose=1)
    return X
