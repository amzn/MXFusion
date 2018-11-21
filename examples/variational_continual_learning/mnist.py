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
from mxnet.io import NDArrayIter


class TaskGenerator:
    def __init__(self, data, batch_size, tasks):
        self.data = data
        self.batch_size = batch_size
        self.tasks = tasks


class SplitTaskGenerator(TaskGenerator):
    def __iter__(self):
        """
        Iterate over tasks
        :return: the next task
        :rtype: NDArrayIter
        """
        for task in self.tasks:
            idx_train_0 = np.where(self.data['train_label'] == task[0])[0]
            idx_train_1 = np.where(self.data['train_label'] == task[1])[0]
            idx_test_0 = np.where(self.data['test_label'] == task[0])[0]
            idx_test_1 = np.where(self.data['test_label'] == task[1])[0]

            x_train = np.vstack((self.data['train_data'][idx_train_0], self.data['train_data'][idx_train_1]))
            y_train = np.hstack((np.ones((idx_train_0.shape[0],)), np.zeros((idx_train_1.shape[0],))))

            x_test = np.vstack((self.data['test_data'][idx_test_0], self.data['test_data'][idx_test_1]))
            y_test = np.hstack((np.ones((idx_test_0.shape[0],)), np.zeros((idx_test_1.shape[0],))))

            batch_size = self.batch_size or x_train.shape[0]
            train_iter = NDArrayIter(x_train, y_train, batch_size, shuffle=True)

            batch_size = self.batch_size or x_test.shape[0]
            test_iter = NDArrayIter(x_test, y_test, batch_size)

            yield train_iter, test_iter
        return


class PermutedTaskGenerator(TaskGenerator):
    def __iter__(self):
        """
        Iterate over tasks
        :return: the next task
        :rtype: NDArrayIter
        """
        for _ in self.tasks:
            x_train = self.data['train_data']
            y_train = self.data['train_label']

            x_test = self.data['test_data']
            y_test = self.data['test_label']

            permutation = np.random.permutation(x_train.shape[1])

            x_train = x_train[:, permutation]
            x_test = x_test[:, permutation]

            # Convert to one hot encodings
            # y_train = np.eye(10)[y_train]
            # y_test = np.eye(10)[y_test]

            batch_size = self.batch_size or x_train.shape[0]
            train_iter = NDArrayIter(x_train, y_train, batch_size, shuffle=True)

            batch_size = self.batch_size or x_test.shape[0]
            test_iter = NDArrayIter(x_test, y_test, batch_size)

            yield train_iter, test_iter
        return
