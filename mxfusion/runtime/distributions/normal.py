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
import numpy as np
from .distribution import DistributionRunTime


class NormalRunTime(DistributionRunTime):

    def __init__(self, mean, variance):
        super(NormalRunTime, self).__init__()
        self.mean = mean
        self.variance = variance

    def log_pdf(self, random_variable):
        logvar = np.log(2 * np.pi) / -2 + mx.nd.log(self.variance) / -2
        logL = mx.nd.broadcast_add(logvar, mx.nd.broadcast_div(mx.nd.square(
            mx.nd.broadcast_minus(random_variable, self.mean)), -2 * self.variance))
        return logL

    def draw_samples(self, num_samples=1):
        out_shape = (num_samples,) + self.mean.shape[1:]
        return mx.nd.broadcast_add(mx.nd.broadcast_mul(mx.nd.random.normal(shape=out_shape, dtype=self.mean.dtype, ctx=self.mean.context), mx.nd.sqrt(self.variance)), self.mean)

    def kl_divergence(self, other_dist):
        raise NotImplementedError
