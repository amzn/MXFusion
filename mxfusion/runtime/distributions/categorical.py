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

from mxnet.ndarray import log_softmax, sum, pick, one_hot, broadcast_to
from mxnet.ndarray import random
from .distribution import DistributionRuntime


class CategoricalRuntime(DistributionRuntime):

    def __init__(self, log_prob, num_classes, one_hot_encoding=False,
                 normalization=True, axis=-1):
        super(CategoricalRuntime, self).__init__()
        self.log_prob = log_prob
        self.num_classes = num_classes
        self.one_hot_encoding = one_hot_encoding
        self.normalization = normalization
        self.axis = axis

    def log_pdf(self, random_variable):
        if self.normalization:
            log_prob = log_softmax(self.log_prob, axis=self.axis)
        else:
            log_prob = self.log_prob

        if self.one_hot_encoding:
            log_pdf = sum(random_variable*log_prob, axis=self.axis)
        else:
            log_pdf = pick(log_prob, index=random_variable, axis=self.axis)
        return log_pdf

    def draw_samples(self, num_samples=1):
        if self.normalization:
            log_prob = log_softmax(self.log_prob, axis=self.axis)
        else:
            log_prob = self.log_prob

        if num_samples != log_prob.shape[0]:
            log_prob = broadcast_to(log_prob, (num_samples,) + log_prob.shape[1:])
        samples = random.multinomial(log_prob, shape=num_samples)
        if self.one_hot_encoding:
            samples = one_hot(samples, depth=self.num_classes)
        return samples
