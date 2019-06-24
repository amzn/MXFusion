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

from mxnet.ndarray import zeros_like, broadcast_axis
from .distribution import DistributionRuntime


class PointMassRuntime(DistributionRuntime):

    def __init__(self, location):
        super(PointMassRuntime, self).__init__()
        self.location = location

    def log_pdf(self, random_variable):
        return zeros_like(random_variable)

    def draw_samples(self, num_samples=1):
        if num_samples != self.location.shape[0]:
            return broadcast_axis(self.location, axis=0, size=num_samples)
        else:
            return self.location

    @property
    def mean(self):
        return self.location

    @property
    def variance(self):
        return zeros_like(self.location)
