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

import mxnet as mx
from mxnet.ndarray import log, log1p, abs, sign
from .distribution import DistributionRuntime


class LaplaceRuntime(DistributionRuntime):

    def __init__(self, location, scale):
        super(LaplaceRuntime, self).__init__()
        self.location = location
        self.scale = scale

    def log_pdf(self, random_variable):
        return -abs(random_variable - self.location)/self.scale - log(2*self.scale)

    def draw_samples(self, num_samples=1):
        # Given a random variable U drawn from the uniform distribution in the interval (-1/2,1/2], the random variable
        # X =\mu - b\, \sgn(U)\, \ln(1 - 2 | U |)
        # has a Laplace distribution with parameters \mu and b

        out_shape = (num_samples,) + self.location.shape[1:]
        U = mx.random.uniform(low=-0.5, high=0.5, shape=out_shape, dtype=self.location.dtype,
                              ctx=self.location.context)
        b_sgn_U = self.scale * sign(U)
        ln_1_2_U = log1p(-2*abs(U))
        return self.location - b_sgn_U*ln_1_2_U

    @property
    def mean(self):
        return self.location

    @property
    def variance(self):
        return 2*self.scale**2
