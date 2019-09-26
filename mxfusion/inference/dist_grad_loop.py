# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from .grad_loop import GradLoop

class DistributedGradLoop(GradLoop):
    """
    The class for the main loop for distributed batch/minibatch loop gradient-based optimization.

    :param data: a list of observed variables from the distributed batch/minibatch loop.
    :type data: [mxnet.ndarray]
    """

    def split_data(self, data):
        import horovod.mxnet as hvd

        if hvd.size() > 1:
            temporaryData = []

            for _, subdata in enumerate(data):
                x = int(subdata.shape[0] / hvd.size())
                y = subdata.shape[0] % hvd.size()
                rank = hvd.rank()
                z = 0 if (rank < y) else 1
                f = 0 if (rank < y + 1) else 1
                start_point = rank*x+rank-f*(rank-y)
                end_point = (rank+1)*x+rank-z*(rank-y+1) + 1
                tempData = mx.nd.slice_axis(subdata, axis=0, begin=start_point, end=end_point)
                temporaryData.append(tempData)

            data = temporaryData

        return data


