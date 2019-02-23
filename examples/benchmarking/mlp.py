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

from mxnet.gluon import Block
from mxnet.gluon.nn import Dense, Sequential


class MLPSequential(Sequential):
    def __init__(self, prefix, network_shape, **kwargs):
        super().__init__(prefix=prefix, **kwargs)

        self.network_shape = network_shape

        with self.name_scope():
            for i in range(1, len(self.network_shape) - 1):
                self.add(Dense(self.network_shape[i], activation="relu", in_units=self.network_shape[i - 1]))

            # Last layer for classification
            self.add(Dense(self.network_shape[-1], in_units=self.network_shape[-2]))


class MLP(Block):
    def __init__(self, prefix, network_shape, **kwargs):
        super().__init__(prefix=prefix, **kwargs)

        self.network_shape = network_shape

        with self.name_scope():
            self.hidden = Sequential()
            for i in range(1, len(network_shape) - 1):
                self.hidden.add(Dense(network_shape[i], activation="relu", in_units=network_shape[i - 1]))

            self.head = Dense(network_shape[-1], in_units=network_shape[-2])

    def forward(self, x):
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)

        return self.head(x)
