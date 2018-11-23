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

import numpy as np
import mxnet as mx
from mxnet.io import NDArrayIter, DataIter, DataBatch
from abc import ABCMeta, abstractmethod
import itertools


class MultiIter(DataIter):
    def __init__(self, iter_list):
        super().__init__()
        self.iterators = [] if iter_list is None else iter_list

    def __next__(self):
        if len(self.iterators) == 0:
            raise StopIteration

        if len(self.iterators) == 1:
            return next(self.iterators[0])

        data = []
        labels = []
        for iterator in self.iterators:
            batch = next(iterator)
            data.append(batch.data)
            labels.append(batch.label)
        return DataBatch(data=mx.nd.concat(*data, axis=0), label=mx.nd.concat(*labels, axis=0), pad=0)

    def __len__(self):
        return len(self.iterators)

    def __getitem__(self, item):
        return self.iterators[item]

    def reset(self):
        for i in self.iterators:
            i.reset()

    @property
    def provide_data(self):
        return list(itertools.chain(map(lambda i: i.provide_data, self.iterators)))

    @property
    def provide_label(self):
        return list(itertools.chain(map(lambda i: i.provide_label, self.iterators)))

    def append(self, iterator):
        if not isinstance(iterator, (DataIter, NDArrayIter)):
            raise ValueError("Expected either a DataIter or NDArray object, received: {}".format(type(iterator)))
        self.iterators.append(iterator)


class Coreset(metaclass=ABCMeta):
    """
    Abstract base class for coresets
    """
    def __init__(self):
        """
        Initialise the coreset
        """
        self.iterator = None
        self.reset()

    @abstractmethod
    def selector(self, data):
        pass

    def update(self, iterator):
        data, labels = iterator.data[0][1].asnumpy(), iterator.label[0][1].asnumpy()
        idx = self.selector(data)
        self.iterator.append(NDArrayIter(data=data[idx, :], label=labels[idx], shuffle=False, batch_size=len(idx)))

        data = np.delete(data, idx, axis=0)
        labels = np.delete(labels, idx, axis=0)
        batch_size = min(iterator.batch_size, data.shape[0])

        return NDArrayIter(data=data, label=labels, shuffle=False, batch_size=batch_size)

    def reset(self):
        self.iterator = MultiIter([])


class Vanilla(Coreset):
    """
    Vanilla coreset that is always size 0
    """
    def __init__(self):
        super().__init__()
        self.coreset_size = 0

    def update(self, iterator):
        return iterator

    def selector(self, data):
        raise NotImplementedError


class Random(Coreset):
    """
    Randomly select from (data, labels) and add to current coreset
    """
    def __init__(self, coreset_size):
        """
        Initialise the coreset
        :param coreset_size: Size of the coreset
        :type coreset_size: int
        """
        super().__init__()
        if coreset_size == 0:
            raise ValueError("Coreset size should be > 0")
        self.coreset_size = coreset_size

    def selector(self, data):
        return np.random.choice(data.shape[0], self.coreset_size, False)


class KCenter(Random):
    """
    Select k centers from (data, labels) and add to current coreset
    """
    def selector(self, data):
        dists = np.full(data.shape[0], np.inf)
        current_id = 0

        # TODO: This looks horribly inefficient
        dists = self.update_distance(dists, data, current_id)
        idx = [current_id]

        for i in range(1, self.coreset_size):
            current_id = np.argmax(dists)
            dists = self.update_distance(dists, data, current_id)
            idx.append(current_id)
        return idx

    @staticmethod
    def update_distance(dists, data, current_id):
        for i in range(data.shape[0]):
            current_dist = np.linalg.norm(data[i, :] - data[current_id, :])
            dists[i] = np.minimum(current_dist, dists[i])
        return dists
