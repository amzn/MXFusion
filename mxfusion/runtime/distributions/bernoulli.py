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
from .distribution import DistributionRuntime


class BernoulliRuntime(DistributionRuntime):

    def __init__(self, prob_true):
        super(BernoulliRuntime, self).__init__()
        self.prob_true = prob_true

    def log_pdf(self, random_variable):
        logL = random_variable * mx.nd.log(self.prob_true) + (1 - random_variable) * mx.nd.log(1 - self.prob_true)
        return logL

    def draw_samples(self, num_samples=1):
        out_shape = (num_samples,) + self.prob_true.shape[1:]
        return mx.random.uniform(low=0, high=1, shape=out_shape, dtype=self.prob_true.dtype) < self.prob_true
