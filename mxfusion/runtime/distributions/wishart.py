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
from mxnet.ndarray import gammaln, log, broadcast_to, ones_like, expand_dims, linalg_makediag, sum, arange, sqrt, linalg_maketrian, diag, broadcast_axis
from mxnet.ndarray import random
from mxnet.ndarray.linalg import potrf, sumlogdiag, trsm, trmm, syrk
from .distribution import DistributionRuntime


class WishartRuntime(DistributionRuntime):

    def __init__(self, degrees_of_freedom, scale):
        super(WishartRuntime, self).__init__()
        self.degrees_of_freedom = degrees_of_freedom
        self.scale = scale
        self._L_scale = potrf(self.scale)

    def log_pdf(self, random_variable):
        n = sum(self.degrees_of_freedom, -1)
        p = self.scale.shape[-1]
        # Constants
        LV = self._L_scale
        LX = potrf(random_variable)
        if random_variable.shape[0] != self.scale.shape[0]:
            LX = broadcast_to(LX, (self.scale.shape[0],) + LX.shape[1:])
        LVinvLX = trsm(LV, LX)
        log_mgamma = p*(p-1)/4*np.log(np.pi) + gammaln((expand_dims(n, axis=-1)-arange(p, dtype=n.dtype, ctx=n.context))/2).sum(-1)

        return -n*p/2*np.log(2) - n*sumlogdiag(LV) - log_mgamma + (n-p-1)*sumlogdiag(LX) \
            - sum((LVinvLX**2).reshape(*(LV.shape[:-2]+(-1,))), -1)/2

    def draw_samples(self, num_samples=1):
        out_shape = (num_samples,)+self.degrees_of_freedom.shape[1:-1]
        LV = self._L_scale
        n = self.degrees_of_freedom
        p = self.scale.shape[-1]
        dtype = self.scale.dtype
        ctx = self.scale.context

        # ChiSquared(k) == Gamma(alpha=k/2, beta=1/2)
        k = arange(0, -p, -1, dtype=dtype, ctx=ctx) + broadcast_axis(n, axis=n.ndim-1, size=p)
        if n.shape[0] < num_samples:
            k = broadcast_axis(k, axis=0, size=num_samples)
            LV = broadcast_axis(LV, axis=0, size=num_samples)
        # The MXNet Gamma sampler is wrong: the beta argument is actually scale.
        c = sqrt(random.gamma(alpha=k/2, beta=ones_like(k)*2, dtype=dtype, ctx=ctx))
        A = linalg_maketrian(random.normal(shape=out_shape+(int((p-1)*p/2),), dtype=dtype, ctx=ctx), offset=-1) + linalg_makediag(c)
        LA = trmm(LV, A)
        return syrk(LA)

    @property
    def mean(self):
        return expand_dims(self.degrees_of_freedom, axis=-1)*self.scale

    @property
    def variance(self):
        V_diag = diag(self.scale, axis1=-2, axis2=-1)
        return expand_dims(self.degrees_of_freedom, axis=-1)*(self.scale**2+syrk(expand_dims(V_diag, axis=-1)))
