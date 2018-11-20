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
                 coreset, batch_size, single_head, ctx):
        self.network_shape = network_shape
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.data_generator = data_generator
        self.coreset = coreset
        self.batch_size = batch_size
        self.single_head = single_head
        self.context = ctx

        # The following are to keep lint happy:
        self.overall_accuracy = None
        self.test_iterators = None
        self.vanilla_model = None
        self.bayesian_model = None
        self.prediction_model = None

        self.reset()

    def reset(self):
        self.coreset.reset()
        self.overall_accuracy = np.array([])
        self.test_iterators = dict()

        model_params = dict(
            network_shape=self.network_shape,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            max_iter=self.num_epochs,
            ctx=self.context
        )

        self.vanilla_model = VanillaNN(**model_params)
        self.bayesian_model = BayesianNN(**model_params)
        self.prediction_model = BayesianNN(**model_params)

    def run(self, verbose=True):
        self.reset()

        # To begin with, set the priors to None.
        # We will in fact use the results of maximum likelihood as the first prior
        priors = None

        for task_id, (train_iterator, test_iterator) in enumerate(self.data_generator):
            print("Task: ", task_id)
            self.test_iterators[task_id] = test_iterator

            # Set the readout head to train_iterator
            head = 0 if self.single_head else task_id

            # Update the coreset, and update the train iterator to remove the coreset data
            train_iterator = self.coreset.update(train_iterator)

            batch_size = train_iterator.provide_label[0].shape[0] if self.batch_size is None else self.batch_size

            # Train network with maximum likelihood to initialize first model
            if task_id == 0:
                # TODO: test_iterator should be val_iter
                print("Training vanilla neural network as starting point")
                self.vanilla_model.train(
                    train_iterator=train_iterator,
                    validation_iterator=test_iterator,
                    task_id=task_id,
                    epochs=5,
                    batch_size=batch_size,
                    verbose=verbose)

                priors = self.vanilla_model.net.collect_params()
                train_iterator.reset()

            # Train on non-coreset data
            # TODO: test_iterator should be val_iter
            print("Training main model")
            self.bayesian_model.train(
                train_iterator=train_iterator,
                validation_iterator=test_iterator,
                task_id=head,
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                priors=priors)

            # Set the priors for the next round of inference to be the current posteriors
            priors = self.bayesian_model.posteriors

            # Incorporate coreset data and make prediction
            acc = self.get_scores()
            print("Accuracies after task {}: [{}]".format(task_id, ", ".join(map("{:.3f}".format, acc))))
            self.overall_accuracy = self.concatenate_results(acc, self.overall_accuracy)

    def get_scores(self):
        acc = []
        prediction_model = self.prediction_model

        if self.single_head:
            if len(self.coreset.iterator) > 0:
                train_iterator = Coreset.merge(self.coreset)
                batch_size = train_iterator.provide_label.shape[0] if (self.batch_size is None) else self.batch_size
                priors = self.bayesian_model.posteriors
                print("Training single-head prediction model")
                prediction_model.train(
                    train_iterator=train_iterator,
                    validation_iterator=None,
                    task_id=0,
                    epochs=self.num_epochs,
                    batch_size=batch_size,
                    priors=priors)
            else:
                print("Using main model as prediction model")
                prediction_model = self.bayesian_model

        for task_id, test_iterator in self.test_iterators.items():
            test_iterator.reset()
            if not self.single_head:
                # TODO: What's the validation data here?
                # TODO: different learning rate and max iter here?
                if len(self.coreset.iterator) > 0:
                    print("Training multi-head prediction model")
                    prediction_model.train(
                        train_iterator=self.coreset.iterator,
                        validation_iterator=None,
                        task_id=task_id,
                        epochs=self.num_epochs,
                        batch_size=self.batch_size,
                        priors=self.bayesian_model.posteriors)
                else:
                    print(f"Using main model as prediction model for task {task_id}")
                    prediction_model = self.bayesian_model

            head = 0 if self.single_head else task_id

            print(f"Generating predictions for task {task_id}")
            predictions = prediction_model.prediction_prob(test_iterator, head)
            predicted_means = np.mean(predictions, axis=0)
            predicted_labels = np.argmax(predicted_means, axis=1)
            test_labels = test_iterator.label[0][1].asnumpy()
            cur_acc = len(np.where(np.abs(predicted_labels - test_labels) < 1e-10)[0]) * 1.0 / test_labels.shape[0]
            acc.append(cur_acc)
        return acc

    @staticmethod
    def concatenate_results(score, all_score):
        if all_score.size == 0:
            all_score = np.reshape(score, (1, -1))
        else:
            new_arr = np.empty((all_score.shape[0], all_score.shape[1] + 1))
            new_arr[:] = np.nan
            new_arr[:, :-1] = all_score
            all_score = np.vstack((new_arr, score))
        return all_score
