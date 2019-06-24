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
from mxnet.ndarray import gammaln, log, log1p, broadcast_to
from mxnet.ndarray import random
from .distribution import DistributionRuntime


class BetaRuntime(DistributionRuntime):

    def __init__(self, alpha, beta):
        super(BetaRuntime, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def log_pdf(self, random_variable):
        log_x = log(random_variable)
        log_1_minus_x = log1p(-random_variable)
        log_beta_ab = gammaln(self.alpha) + gammaln(self.beta) - gammaln(self.alpha + self.beta)
        return (self.alpha - 1) * log_x + (self.beta - 1) * log_1_minus_x - log_beta_ab

    def draw_samples(self, num_samples=1):
        if num_samples != self.alpha.shape[0]:
            alpha = broadcast_to(self.alpha, (num_samples,) + self.alpha.shape[1:])
        else:
            alpha = self.alpha
        if num_samples != self.beta.shape[0]:
            beta = broadcast_to(self.beta, (num_samples,) + self.beta.shape[1:])
        else:
            beta = self.beta
        # Sample X from Gamma(a, 1)
        x = random.gamma(alpha=alpha, beta=mx.nd.ones_like(alpha, dtype=alpha.dtype, ctx=alpha.context),
                         dtype=alpha.dtype, ctx=alpha.context)

        # Sample Y from Gamma(b, 1)
        y = random.gamma(alpha=beta, beta=mx.nd.ones_like(beta, dtype=beta.dtype, ctx=beta.context),
                         dtype=beta.dtype, ctx=beta.context)

        return x / (x + y)

    @property
    def mean(self):
        return self.alpha/(self.alpha+self.beta)

    @property
    def variance(self):
        return self.alpha*self.beta/((self.alpha+self.beta)**2 * (self.alpha+self.beta+1))
