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


from mxnet.ndarray import gammaln, log, broadcast_to, ones_like, expand_dims, linalg_makediag, sum
from mxnet.ndarray import random
from .distribution import DistributionRuntime


class DirichletRuntime(DistributionRuntime):

    def __init__(self, alpha):
        super(DirichletRuntime, self).__init__()
        self.alpha = alpha

    def log_pdf(self, random_variable):
        return gammaln(sum(self.alpha, axis=-1)) + \
            sum(-gammaln(self.alpha) + (self.alpha-1)*log(random_variable), axis=-1)

    def draw_samples(self, num_samples=1):
        if num_samples != self.alpha.shape[0]:
            alpha = broadcast_to(self.alpha, (num_samples,) + self.alpha.shape[1:])
        else:
            alpha = self.alpha
        y = random.gamma(alpha=alpha, beta=ones_like(alpha), dtype=alpha.dtype, ctx=alpha.context)
        return y / sum(y, axis=-1, keepdims=True)

    @property
    def mean(self):
        return self.alpha/sum(self.alpha, axis=-1, keepdims=True)

    @property
    def variance(self):
        alpha_sum = sum(self.alpha, axis=-1, keepdims=True)
        alpha_norm = self.alpha/alpha_sum
        return alpha_norm*(1-alpha_norm)/(alpha_sum+1)

    @property
    def covariance(self):
        alpha_sum = sum(self.alpha, axis=-1, keepdims=True)
        alpha_norm = self.alpha/alpha_sum
        return (linalg_makediag(alpha_norm)-expand_dims(alpha_norm, axis=-1)*expand_dims(alpha_norm, axis=-2))/(alpha_sum+1)
