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
from mxnet.gluon import Trainer, ParameterDict, Block
from mxnet.gluon.contrib.nn import Concurrent
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import HybridSequential, Dense, Sequential
from mxnet.initializer import Xavier
from mxnet.metric import Accuracy

from mxfusion import Model, Variable
from mxfusion.components import MXFusionGluonFunction
from mxfusion.components.distributions import Normal, Categorical
from mxfusion.inference import BatchInferenceLoop, create_Gaussian_meanfield, GradBasedInference, \
    StochasticVariationalInference, VariationalPosteriorForwardSampling
from abc import ABC, abstractmethod


class BaseNN(ABC):
    prefix = None

    def __init__(self, network_shape, learning_rate, optimizer, max_iter, ctx):
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

    @property
    def single_head(self):
        if isinstance(self.network_shape[-1], int):
            return True
        if isinstance(self.network_shape[-1], (tuple, list)):
            return False
        raise ValueError("Unsupported network shape")

    @property
    def num_heads(self):
        return 1 if self.single_head else len(self.network_shape[-1])

    class MLP(Block):
        def __init__(self, prefix, network_shape, single_head, **kwargs):
            super().__init__(prefix=prefix, **kwargs)
            # self.hidden_layers = []
            self.single_head = single_head

            with self.name_scope():
                self.hidden = Sequential()
                for i in range(1, len(network_shape) - 1):
                    self.hidden.add(Dense(network_shape[i], activation="relu", in_units=network_shape[i - 1]))

                # for i in range(1, len(network_shape) - 1):
                #     self.hidden_layers.append(
                #         Dense(network_shape[i], activation="relu", in_units=network_shape[i - 1]))

                self.dense1 = Dense(64, activation="relu")

                if single_head:
                    self.head = Dense(network_shape[-1], in_units=network_shape[-2])
                else:
                    self.concurrent = Concurrent()
                    # self.heads = []
                    for label_shape in network_shape[-1]:
                        self.concurrent.add(Dense(label_shape, in_units=network_shape[-2]))
                        # self.heads.append(Dense(label_shape, in_units=network_shape[-2]))

        def forward(self, x):
            for i in range(len(self.hidden)):
                x = self.hidden[i](x)

            # for layer in self.hidden_layers:
            #     x = layer(x)

            if self.single_head:
                return self.head(x)
            else:
                return tuple(map(lambda h: h(x), self.concurrent))
                # return tuple(map(lambda h: h(x), self.heads))

    def create_net(self):
        # Create net
        self.net = self.MLP(self.prefix, self.network_shape, self.single_head)
        self.net.initialize(Xavier(magnitude=2.34), ctx=self.ctx)

        # self.net = HybridSequential(prefix=self.prefix)
        # with self.net.name_scope():
        #     for i in range(1, len(self.network_shape) - 1):
        #         self.net.add(Dense(self.network_shape[i], activation="relu", in_units=self.network_shape[i - 1]))
        #
        #     # Last layer for classification - one per head for multi-head networks
        #     if self.single_head:
        #         self.net.add(Dense(self.network_shape[-1], in_units=self.network_shape[-2]))
        #     else:
        #         for label_shape in self.network_shape[-1]:
        #             self.net.add(Dense(label_shape, in_units=self.network_shape[-2]))
        # self.net.initialize(Xavier(magnitude=2.34), ctx=self.ctx)

    def forward(self, data):
        # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
        data = mx.nd.flatten(data).as_in_context(self.ctx)
        output = self.net(data)
        return output

    def evaluate_accuracy(self, data_iterator):
        """
        Evaluate the accuracy of the model on the given data iterator
        :param data_iterator: data iterator
        :return: accuracy
        :rtype: float
        """
        acc = Accuracy()
        for i, batch in enumerate(data_iterator):
            output = self.forward(batch.data[0])
            labels = batch.label[0].as_in_context(self.ctx)
            predictions = mx.nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=labels)
        return acc.get()[1]

    @abstractmethod
    def train(self, train_iterator, validation_iterator, head, batch_size, epochs, priors=None, verbose=True):
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

    def train(self, train_iterator, validation_iterator, head, batch_size, epochs, priors=None, verbose=True):
        trainer = Trainer(self.net.collect_params(), self.optimizer, dict(learning_rate=self.learning_rate))

        num_examples = 0
        for epoch in range(epochs):
            cumulative_loss = 0
            for i, batch in enumerate(train_iterator):
                with mx.autograd.record():
                    output = self.forward(batch.data[0].as_in_context(self.ctx))[head]
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
            self.print_status(epoch + 1, cumulative_loss / num_examples, train_accuracy, validation_accuracy)


class BayesianNN(BaseNN):
    prefix = 'bayesian_'

    def __init__(self, network_shape, learning_rate, optimizer, max_iter, ctx):
        super().__init__(network_shape, learning_rate, optimizer, max_iter, ctx)
        self.create_model()

    def create_model(self):
        self.model = Model(verbose=True)
        self.model.N = Variable()
        self.model.f = MXFusionGluonFunction(self.net, num_outputs=self.num_heads, broadcastable=False)
        self.model.x = Variable(shape=(self.model.N, self.network_shape[0]))

        if self.single_head:
            self.model.r = self.model.f(self.model.x)
            self.model.y = Categorical.define_variable(
                log_prob=self.model.r, shape=(self.model.N, 1), num_classes=self.network_shape[-1])
            self.create_prior_variables(self.model.r)
        else:
            for head, label_shape in enumerate(self.network_shape[-1]):
                r = self.model.f(self.model.x)[head] if self.num_heads > 1 else self.model.f(self.model.x)
                setattr(self.model, f'r{head}', r)
                y = Categorical.define_variable(log_prob=r, shape=(self.model.N, 1), num_classes=label_shape)
                setattr(self.model, f'y{head}', y)
                # TODO the statement below could probably be done only for the first head, since they all share the same
                # factor parameters
                self.create_prior_variables(r)

    def create_prior_variables(self, r):
        for v in r.factor.parameters.values():
            # First check that the variables haven't already been created (in multi-head case)
            if getattr(self.model, v.inherited_name + "_mean", None) is not None:
                continue
            if getattr(self.model, v.inherited_name + "_variance", None) is not None:
                continue

            means = Variable(shape=v.shape)
            variances = Variable(shape=v.shape)
            setattr(self.model, v.inherited_name + "_mean", means)
            setattr(self.model, v.inherited_name + "_variance", variances)
            v.set_prior(Normal(mean=means, variance=variances))

    # noinspection PyUnresolvedReferences
    def get_net_parameters(self, head):
        if self.single_head:
            r = self.model.r
        else:
            r = getattr(self.model, f'r{head}')
        return r.factor.parameters

    # noinspection PyUnresolvedReferences
    def train(self, train_iterator, validation_iterator, head, batch_size, epochs, priors=None, verbose=True):
        for i, batch in enumerate(train_iterator):
            if i > 0:
                raise NotImplementedError("Currently not supported for more than one batch of data. "
                                          "Please switch to using the MinibatchInferenceLoop")

            data = mx.nd.flatten(batch.data[0]).as_in_context(self.ctx)
            labels = mx.nd.expand_dims(batch.label[0], axis=-1).as_in_context(self.ctx)

            # pass some data to initialise the net
            # self.net(data[:1])

            # TODO: Would rather have done this before!
            # self.create_model()

            if self.single_head:
                observed = [self.model.x, self.model.y]
                kwargs = dict(y=labels, x=data)
            else:
                observed = [self.model.x, getattr(self.model, f"y{head}")]
                kwargs = {'x': data, f'y{head}': labels}

            q = create_Gaussian_meanfield(model=self.model, observed=observed)
            alg = StochasticVariationalInference(num_samples=5, model=self.model, posterior=q, observed=observed)
            self.inference = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
            self.inference.initialize(**kwargs)

            for v in self.get_net_parameters(head).values():
                v_name_mean = v.inherited_name + "_mean"
                v_name_variance = v.inherited_name + "_variance"

                if priors is None or (v_name_mean not in priors and v_name_variance not in priors):
                    means = self.prior_mean(shape=v.shape)
                    variances = self.prior_variance(shape=v.shape)
                elif isinstance(priors, ParameterDict):
                    # This is a maximum likelihood estimate
                    short_name = v.inherited_name.partition(self.prefix)[-1]
                    means = priors.get(short_name).data()
                    variances = self.prior_variance(shape=v.shape)
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
                               verbose=False, callback=self.print_status, **kwargs)

    # noinspection PyUnresolvedReferences
    @property
    def posteriors(self):
        q = self.inference.inference_algorithm.posterior
        posteriors = dict()
        if self.single_head:
            for v_name, v in self.model.r.factor.parameters.items():
                posteriors[v.inherited_name + "_mean"] = self.inference.params[q[v.uuid].factor.mean].asnumpy()
                posteriors[v.inherited_name + "_variance"] = self.inference.params[q[v.uuid].factor.variance].asnumpy()
        else:
            for head in range(self.num_heads):
                for v in self.get_net_parameters(head).values():
                    posteriors[v.inherited_name + "_mean"] = self.inference.params[q[v.uuid].factor.mean].asnumpy()
                    posteriors[v.inherited_name + "_variance"] = \
                        self.inference.params[q[v.uuid].factor.variance].asnumpy()
        return posteriors

    # noinspection PyUnresolvedReferences
    def prediction_prob(self, test_iter, head):
        if self.inference is None:
            raise RuntimeError("Model not yet learnt")

        for i, batch in enumerate(test_iter):
            if i > 0:
                raise NotImplementedError("Currently not supported for more than one batch of data. "
                                          "Please switch to using the MinibatchInferenceLoop")

            data = mx.nd.flatten(batch.data[0]).as_in_context(self.ctx)

            if self.single_head:
                r = self.model.r
            else:
                r = getattr(self.model, f'r{head}')

            prediction_inference = VariationalPosteriorForwardSampling(
                10, [self.model.x], self.inference, [r])
            res = prediction_inference.run(x=mx.nd.array(data))
            return res[0].asnumpy()

    @staticmethod
    def prior_mean(shape):
        return mx.nd.zeros(shape=shape)

    @staticmethod
    def prior_variance(shape):
        return mx.nd.ones(shape=shape) * 3
