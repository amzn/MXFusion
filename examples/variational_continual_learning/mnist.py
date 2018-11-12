import mxnet as mx
import numpy as np


class SplitMnistGenerator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_tasks = 5

    def __iter__(self):
        for i in range(self.num_tasks):
            idx_train_0 = np.where(self.data['train_label'] == i * 2)[0]
            idx_train_1 = np.where(self.data['train_label'] == i * 2 + 1)[0]
            idx_test_0 = np.where(self.data['test_label'] == i * 2)[0]
            idx_test_1 = np.where(self.data['test_label'] == i * 2 + 1)[0]

            x_train = np.vstack((self.data['train_data'][idx_train_0], self.data['train_data'][idx_train_1]))
            y_train = np.vstack((np.ones((idx_train_0.shape[0], 1)), -np.ones((idx_train_1.shape[0], 1))))

            x_test = np.vstack((self.data['test_data'][idx_test_0], self.data['test_data'][idx_test_1]))
            y_test = np.vstack((np.ones((idx_test_0.shape[0], 1)), -np.ones((idx_test_1.shape[0], 1))))

            batch_size = x_train.shape[0] if self.batch_size is None else self.batch_size
            train_iter = mx.io.NDArrayIter(x_train, y_train, batch_size, shuffle=True)

            batch_size = x_test.shape[0] if self.batch_size is None else self.batch_size
            test_iter = mx.io.NDArrayIter(x_test, y_test, batch_size)

            yield train_iter, test_iter
        return
