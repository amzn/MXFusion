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
import mxnet as mx
from mxnet.gluon import Trainer, ParameterDict
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import HybridSequential, Dense
from mxnet.initializer import Xavier
from mxnet.metric import Accuracy

from mxfusion import Model, Variable
from mxfusion.components import MXFusionGluonFunction
from mxfusion.components.distributions import Normal, Categorical
from mxfusion.inference import BatchInferenceLoop, create_Gaussian_meanfield, GradBasedInference, \
    StochasticVariationalInference, VariationalPosteriorForwardSampling
import numpy as np
from abc import ABC, abstractmethod


class BaseNN(ABC):
    prefix = None

    def __init__(self, network_shape, learning_rate, optimizer, max_iter, ctx):
        self.task_idx = mx.nd.array([-1], dtype=np.float32)
        self.model = None
        self.network_shape = network_shape
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.loss = None
        self.ctx = ctx
        self.model = None
        self.net = None
        self.inference = None
        self.create_net()
        self.loss = SoftmaxCrossEntropyLoss()

    def create_net(self):
        # Create net
        self.net = HybridSequential(prefix=self.prefix)
        with self.net.name_scope():
            for i in range(1, len(self.network_shape) - 1):
                self.net.add(Dense(self.network_shape[i], activation="relu", in_units=self.network_shape[i - 1]))
            #  Last layer for classification
            self.net.add(Dense(self.network_shape[-1], in_units=self.network_shape[-2]))
        self.net.initialize(Xavier(magnitude=2.34), ctx=self.ctx)

    def forward(self, data):
        # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
        data = mx.nd.flatten(data).as_in_context(self.ctx)
        output = self.net(data)
        return output

    def evaluate_accuracy(self, data_iterator):
        acc = Accuracy()
        for i, batch in enumerate(data_iterator):
            output = self.forward(batch.data[0])
            labels = batch.label[0].as_in_context(self.ctx)
            predictions = mx.nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=labels)
        return acc.get()[1]

    @abstractmethod
    def train(self, train_iterator, validation_iterator, task_id, batch_size, epochs, priors=None, verbose=True):
        raise NotImplementedError

    def prediction_prob(self, test_iter, task_idx):
        # TODO task_idx??
        prob = self.model.predict(test_iter)
        return prob

    def get_weights(self):
        params = self.net.collect_params()
        return params

    @staticmethod
    def print_status(epoch, loss, train_accuracy=float("nan"), validation_accuracy=float("nan")):
        print(f"Epoch {epoch:4d}. Loss: {loss:8.2f}, "
              f"Train accuracy {train_accuracy:.3f}, Validation accuracy {validation_accuracy:.3f}")


class VanillaNN(BaseNN):
    prefix = 'vanilla_'

    def train(self, train_iterator, validation_iterator, task_id, batch_size, epochs, priors=None, verbose=True):
        trainer = Trainer(self.net.collect_params(), self.optimizer, dict(learning_rate=self.learning_rate))

        num_examples = 0
        for epoch in range(epochs):
            cumulative_loss = 0
            for i, batch in enumerate(train_iterator):
                with mx.autograd.record():
                    output = self.forward(batch.data[0])
                    labels = batch.label[0].as_in_context(self.ctx)
                    loss = self.loss(output, labels)
                loss.backward()
                trainer.step(batch_size=batch_size, ignore_stale_grad=True)
                cumulative_loss += mx.nd.sum(loss).asscalar()
                num_examples += len(labels)

            train_iterator.reset()
            validation_iterator.reset()
            train_accuracy = self.evaluate_accuracy(train_iterator)
            validation_accuracy = self.evaluate_accuracy(validation_iterator)
            self.print_status(epoch, cumulative_loss / num_examples, train_accuracy, validation_accuracy)


class BayesianNN(BaseNN):
    prefix = 'bayesian_'

    def __init__(self, network_shape, learning_rate, optimizer, max_iter, ctx):
        super().__init__(network_shape, learning_rate, optimizer, max_iter, ctx)
        self.create_model()

    def create_model(self):
        self.model = Model()
        self.model.N = Variable()
        self.model.f = MXFusionGluonFunction(self.net, num_outputs=1, broadcastable=False)
        self.model.x = Variable(shape=(self.model.N, self.network_shape[0]))
        self.model.r = self.model.f(self.model.x)
        self.model.y = Categorical.define_variable(log_prob=self.model.r, shape=(self.model.N, 1), num_classes=2)

        for v in self.model.r.factor.parameters.values():
            means = Variable(shape=v.shape)
            variances = Variable(shape=v.shape)
            setattr(self.model, v.inherited_name + "_mean", means)
            setattr(self.model, v.inherited_name + "_variance", variances)
            v.set_prior(Normal(mean=means, variance=variances))

    def train(self, train_iterator, validation_iterator, task_id, batch_size, epochs, priors=None, verbose=True):
        for i, batch in enumerate(train_iterator):
            if i > 0:
                raise NotImplementedError("Currently not supported for more than one batch of data. "
                                          "Please switch to using the MinibatchInferenceLoop")

            data = mx.nd.flatten(batch.data[0]).as_in_context(self.ctx)
            labels = mx.nd.expand_dims(batch.label[0], axis=-1).as_in_context(self.ctx)

            # pass some data to initialise the net
            self.net(data[:1])

            # TODO: Would rather have done this before!
            # self.create_model()

            observed = [self.model.x, self.model.y]
            q = create_Gaussian_meanfield(model=self.model, observed=observed)
            alg = StochasticVariationalInference(num_samples=5, model=self.model, posterior=q, observed=observed)
            self.inference = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
            self.inference.initialize(y=labels, x=data)

            for v in self.model.r.factor.parameters.values():
                v_name_mean = v.inherited_name + "_mean"
                v_name_variance = v.inherited_name + "_variance"

                if priors is None:
                    means = mx.nd.zeros(shape=v.shape)
                    variances = mx.nd.ones(shape=v.shape) * 3
                elif isinstance(priors, ParameterDict):
                    # This is a maximum likelihood estimate
                    short_name = v.inherited_name.partition(self.prefix)[-1]
                    means = priors.get(short_name).data()
                    variances = mx.nd.ones(shape=v.shape) * 3
                else:
                    # Use posteriors from previous round of inference
                    means = priors[v_name_mean]
                    variances = priors[v_name_variance]

                mean_prior = getattr(self.model, v_name_mean)
                variance_prior = getattr(self.model, v_name_variance)

                # v.set_prior(Normal(mean=mean_prior, variance=variance_prior))

                self.inference.params[mean_prior] = means
                self.inference.params[variance_prior] = variances

                # Indicate that we don't want to perform inference over the priors
                self.inference.params.param_dict[mean_prior]._grad_req = 'null'
                self.inference.params.param_dict[variance_prior]._grad_req = 'null'

            self.inference.run(max_iter=self.max_iter, learning_rate=self.learning_rate,
                               x=data, y=labels, verbose=False, callback=self.print_status)

    @property
    def posteriors(self):
        q = self.inference.inference_algorithm.posterior
        posteriors = dict()
        for v_name, v in self.model.r.factor.parameters.items():
            posteriors[v.inherited_name + "_mean"] = self.inference.params[q[v.uuid].factor.mean].asnumpy()
            posteriors[v.inherited_name + "_variance"] = self.inference.params[q[v.uuid].factor.variance].asnumpy()
        return posteriors

    def prediction_prob(self, test_iter, task_idx):
        if self.inference is None:
            raise RuntimeError("Model not yet learnt")

        for i, batch in enumerate(test_iter):
            if i > 0:
                raise NotImplementedError("Currently not supported for more than one batch of data. "
                                          "Please switch to using the MinibatchInferenceLoop")

            data = mx.nd.flatten(batch.data[0]).as_in_context(self.ctx)
            N, D = map(lambda x: mx.nd.array([x], ctx=self.ctx), data.shape)

            prediction_inference = VariationalPosteriorForwardSampling(
                10, [self.model.x], self.inference, [self.model.r])
            res = prediction_inference.run(x=mx.nd.array(data))
            return res[0].asnumpy()
