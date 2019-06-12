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

from mxnet.ndarray import gammaln, log, broadcast_to
from mxnet.ndarray import random
from .distribution import DistributionRuntime


class GammaRuntime(DistributionRuntime):

    def __init__(self, alpha, beta):
        super(GammaRuntime, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def log_pdf(self, random_variable):
        g_alpha = gammaln(self.alpha)
        p1 = (self.alpha - 1.) * log(random_variable)
        return (p1 - self.beta * random_variable) - (g_alpha - self.alpha * log(self.beta))

    def draw_samples(self, num_samples=1):
        if num_samples != self.alpha.shape[0]:
            alpha = broadcast_to(self.alpha, (num_samples,) + self.alpha.shape[1:])
        else:
            alpha = self.alpha
        if num_samples != self.beta.shape[0]:
            beta = broadcast_to(self.beta, (num_samples,) + self.beta.shape[1:])
        else:
            beta = self.beta
        return random.gamma(alpha=alpha, beta=1/beta, dtype=alpha.dtype, ctx=alpha.context)

    @property
    def mean(self):
        return self.alpha/self.beta

    @property
    def variance(self):
        return self.alpha/self.beta**2
