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

from examples.variational_continual_learning.models import VanillaNN, BayesianNN
from examples.variational_continual_learning.coresets import Coreset


class Experiment:
    def __init__(self, network_shape, num_epochs, learning_rate, optimizer, data_generator,
                 coreset, batch_size, single_head, ctx, verbose):
        self.network_shape = network_shape
        self.original_network_shape = network_shape  # Only used when resetting
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.data_generator = data_generator
        self.coreset = coreset
        self.batch_size = batch_size
        self.single_head = single_head
        self.context = ctx
        self.verbose = verbose

        # The following are to keep lint happy:
        self.overall_accuracy = None
        self.test_iterators = None
        self.vanilla_model = None
        self.bayesian_model = None

        self.task_ids = []

    @property
    def model_params(self):
        return dict(
            network_shape=self.network_shape,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            max_iter=self.num_epochs,
            ctx=self.context,
            verbose=self.verbose
        )

    def reset(self):
        self.coreset.reset()
        self.network_shape = self.original_network_shape
        self.overall_accuracy = np.array([])
        self.test_iterators = dict()
        self.task_ids = []

        print("Creating Vanilla Model")
        self.vanilla_model = VanillaNN(**self.model_params)

    def new_task(self, task):
        if self.single_head and self.bayesian_model is not None:
            return

        if len(self.task_ids) > 0:
            self.network_shape = self.network_shape[0:-1] + (self.network_shape[-1] + (task.number_of_classes,),)

        self.task_ids.append(task.task_id)

        # TODO: Would be nice if we could use the same object here
        self.bayesian_model = BayesianNN(**self.model_params)

    def run(self):
        self.reset()

        # To begin with, set the priors to None.
        # We will in fact use the results of maximum likelihood as the first prior
        priors = None

        for task in self.data_generator:
            print("Task: ", task.task_id)
            self.test_iterators[task.task_id] = task.test_iterator

            # Set the readout head to train_iterator
            head = 0 if self.single_head else task.task_id

            # Update the coreset, and update the train iterator to remove the coreset data
            train_iterator = self.coreset.update(task.train_iterator)

            label_shape = train_iterator.provide_label[0].shape
            batch_size = label_shape[0] if self.batch_size is None else self.batch_size

            # Train network with maximum likelihood to initialize first model
            if len(self.task_ids) == 0:
                print("Training non-Bayesian neural network as starting point")
                self.vanilla_model.train(
                    train_iterator=train_iterator,
                    validation_iterator=task.test_iterator,
                    head=head,
                    epochs=5,
                    batch_size=batch_size)

                priors = self.vanilla_model.net.collect_params()
                train_iterator.reset()

            self.new_task(task)

            # Train on non-coreset data
            print("Training main model")
            self.bayesian_model.train(
                train_iterator=train_iterator,
                validation_iterator=task.test_iterator,
                head=head,
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                priors=priors)

            # Set the priors for the next round of inference to be the current posteriors
            priors = self.bayesian_model.posteriors
            # print("Number of variables in priors: {}".format(len(priors.items())))

            # Incorporate coreset data and make prediction
            acc = self.get_scores()
            print("Accuracies after task {}: [{}]".format(task.task_id, ", ".join(map("{:.3f}".format, acc))))
            self.overall_accuracy = concatenate_results(acc, self.overall_accuracy)

    def get_coreset(self, task_id):
        """
        For multi-headed models gets the coreset for the given task id.
        For single-headed models this will return a merged coreset
        :param task_id: The task id
        :return: iterator for the coreset
        """
        if self.single_head:
            # TODO: Cache the results if this is expensive?
            iterator = Coreset.merge(self.coreset).iterator
        else:
            iterator = self.coreset.iterator

        if len(iterator) > 0:
            return iterator[task_id]
        return None

    def fine_tune(self, task_id):
        """
        Fine tune the latest trained model using the coreset(s)
        :param task_id: the task id
        :return: the fine tuned prediction model
        """
        coreset_iterator = self.get_coreset(task_id)

        if coreset_iterator is None:
            print("Empty coreset: Using main model as prediction model for task {}".format(task_id))
            return self.bayesian_model

        coreset_iterator.reset()
        batch_size = coreset_iterator.provide_label[0].shape[0]
        prediction_model = BayesianNN(**self.model_params)

        priors = self.bayesian_model.posteriors
        print("Number of variables in priors: {}".format(len(priors)))

        print("Fine tuning prediction model for task {}".format(task_id))
        prediction_model.train(
            train_iterator=coreset_iterator,
            validation_iterator=None,
            head=task_id,
            epochs=self.num_epochs,
            batch_size=batch_size,
            priors=priors)

        return prediction_model

    def get_scores(self):
        scores = []
        # TODO: different learning rate and max iter here?

        for task_id, test_iterator in self.test_iterators.items():
            test_iterator.reset()

            head = 0 if self.single_head else task_id
            prediction_model = self.fine_tune(task_id)

            print("Generating predictions for task {}".format(task_id))
            predictions = prediction_model.prediction_prob(test_iterator, head)
            predicted_means = np.mean(predictions, axis=0)
            predicted_labels = np.argmax(predicted_means, axis=1)
            test_labels = test_iterator.label[0][1].asnumpy()
            mt = test_labels.shape[0]
            score = len(np.where(np.abs(predicted_labels[:mt] - test_labels) < 1e-10)[0]) * 1.0 / mt
            scores.append(score)
        return scores


def concatenate_results(score, all_score):
    if all_score.size == 0:
        all_score = np.reshape(score, (1, -1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1] + 1))
        new_arr[:] = np.nan
        new_arr[:, :-1] = all_score
        all_score = np.vstack((new_arr, score))
    return all_score
