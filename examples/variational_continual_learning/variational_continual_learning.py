import numpy as np
import gzip
import sys

import mxfusion as mf
import mxnet as mx

import matplotlib.pyplot as plt

from examples.variational_continual_learning.mnist import SplitMnistGenerator
from examples.variational_continual_learning.nets import VanillaNN, MeanFieldNN
from examples.variational_continual_learning.coresets import Random, KCenter, Coreset

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

# Set the compute context, GPU is available otherwise CPU
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()


def set_seeds(seed=42):
    mx.random.seed(seed)
    np.random.seed(seed)


def plot(filename, vcl, rand_vcl, kcen_vcl):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    plt.plot(np.arange(len(vcl))+1, vcl, label='VCL', marker='o')
    plt.plot(np.arange(len(rand_vcl))+1, rand_vcl, label='VCL + Random Coreset', marker='o')
    plt.plot(np.arange(len(kcen_vcl))+1, kcen_vcl, label='VCL + K-center Coreset', marker='o')
    ax.set_xticks(range(1, len(vcl)+1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')
    plt.close()


class Experiment:
    def __init__(self, network_shape, num_epochs, data_generator,
                 coreset_func, batch_size, single_head):
        self.network_shape = network_shape
        self.num_epochs = num_epochs
        self.data_generator = data_generator
        self.coresets = dict((i, coreset_func()) for i in range(gen.num_tasks))
        self.batch_size = batch_size
        self.single_head = single_head
        self.overall_accuracy = np.array([])
        self.x_test_sets = []
        self.y_test_sets = []

    def run(self):
        self.x_test_sets = []
        self.y_test_sets = []

        for task_id, (train_iter, test_iter) in enumerate(self.data_generator):
            self.x_test_sets.append(test_iter.data[0][1])
            self.y_test_sets.append(test_iter.label[0][1])

            # Set the readout head to train_iter
            head = 0 if self.single_head else task_id

            mean_field_weights = None
            mean_field_variances = None

            # Train network with maximum likelihood to initialize first model
            if task_id == 0:
                vanilla_model = VanillaNN(nn_shape)
                vanilla_model.train(train_iter, task_id, self.num_epochs, self.batch_size)
                mean_field_weights = vanilla_model.get_weights()

            # Train on non-coreset data
            mean_field_model = MeanFieldNN(
                nn_shape, prior_means=mean_field_weights, prior_log_variances=mean_field_variances)
            mean_field_model.train(train_iter, head, self.num_epochs, self.batch_size)
            mean_field_weights, mean_field_variances = mean_field_model.get_weights()

            # Incorporate coreset data and make prediction
            acc = self.get_scores(mean_field_model)
            self.overall_accuracy = self.concatenate_results(acc, self.overall_accuracy)

    def get_scores(self, model):
        mf_weights, mf_variances = model.get_weights()
        acc = []
        final_model = None

        if self.single_head:
            if len(self.coresets) > 0:
                x_train, y_train = Coreset.merge(self.coreset)
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                final_model = MeanFieldNN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0],
                                          prev_means=mf_weights, prev_log_variances=mf_variances)
                final_model.train(x_train, y_train, 0, no_epochs, bsize)
            else:
                final_model = model

        for i in range(len(x_testsets)):
            if not single_head:
                if len(x_coresets) > 0:
                    x_train, y_train = x_coresets[i], y_coresets[i]
                    bsize = x_train.shape[0] if (batch_size is None) else batch_size
                    final_model = MeanFieldNN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0],
                                              prev_means=mf_weights, prev_log_variances=mf_variances)
                    final_model.train(x_train, y_train, i, no_epochs, bsize)
                else:
                    final_model = model

            head = 0 if single_head else i
            x_test, y_test = x_testsets[i], y_testsets[i]

            pred = final_model.prediction_prob(x_test, head)
            pred_mean = np.mean(pred, axis=0)
            pred_y = np.argmax(pred_mean, axis=1)
            y = np.argmax(y_test, axis=1)
            cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
            acc.append(cur_acc)

            if len(x_coresets) > 0 and not single_head:
                final_model.close_session()

        if len(x_coresets) > 0 and single_head:
            final_model.close_session()

        return acc

    @staticmethod
    def concatenate_results(score, all_score):
        if all_score.size == 0:
            all_score = np.reshape(score, (1, -1))
        else:
            new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
            new_arr[:] = np.nan
            new_arr[:,:-1] = all_score
            all_score = np.vstack((new_arr, score))
        return all_score


if __name__ == "__main__":
    # Load data
    data = mx.test_utils.get_mnist()
    input_dim = np.prod(data['train_data'][0].shape)  # Note the data will get flattened later
    gen = SplitMnistGenerator(data, batch_size=None)

    nn_shape = (input_dim, 256, 256, 2)  # binary classification
    experiments = dict(
        vanilla=dict(coreset_func=lambda: Random(coreset_size=0),
                     network_shape=nn_shape, num_epochs=120, single_head=False),
        random=dict(coreset_func=lambda: Random(coreset_size=40),
                    network_shape=nn_shape, num_epochs=120, single_head=False),
        k_center=dict(coreset_func=lambda: KCenter(coreset_size=40),
                      network_shape=nn_shape, num_epochs=120, single_head=False)
    )

    # Run experiments
    for name, params in experiments.items():
        print("Running experiment", name)
        set_seeds()
        experiment = Experiment(batch_size=None, data_generator=gen, **params)
        experiment.run()
        print(experiment.overall_accuracy)
