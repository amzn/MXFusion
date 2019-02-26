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
from mxnet.initializer import Xavier
from mxnet.metric import Accuracy

from mxfusion import Model, Variable
from mxfusion.components import MXFusionGluonFunction
from mxfusion.components.distributions import Normal, Categorical
from mxfusion.inference import BatchInferenceLoop, create_Gaussian_meanfield, GradBasedInference, \
    StochasticVariationalInference, VariationalPosteriorForwardSampling
from mxfusion.components.variables import add_sample_dimension
from abc import ABC, abstractmethod

from .mlp import MLP


class BaseNN(ABC):
    prefix = None

    def __init__(self, network_shape, learning_rate, optimizer, max_iter, ctx, verbose):
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
        self.verbose = verbose

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

    def create_net(self):
        # Create net
        self.net = MLP(self.prefix, self.network_shape, self.single_head)
        self.net.initialize(Xavier(magnitude=2.34), ctx=self.ctx)

    def forward(self, data):
        # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
        data = mx.nd.flatten(data).as_in_context(self.ctx)
        output = self.net(data)
        return output

    def evaluate_accuracy(self, data_iterator, head=0):
        """
        Evaluate the accuracy of the model on the given data iterator
        :param data_iterator: data iterator
        :param head: the head of the network (for multi-head models)
        :return: accuracy
        :rtype: float
        """
        acc = Accuracy()
        for i, batch in enumerate(data_iterator):
            if self.single_head:
                output = self.forward(batch.data[0])
            else:
                output = self.forward(batch.data[0])[head]

            labels = batch.label[0].as_in_context(self.ctx)
            predictions = mx.nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=labels)
        return acc.get()[1]

    @abstractmethod
    def train(self, train_iterator, validation_iterator, head, batch_size, epochs, priors=None):
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
        print("Epoch {:4d}. Loss: {:8.2f}, Train accuracy {:.3f}, Validation accuracy {:.3f}"
              .format(epoch, loss, train_accuracy, validation_accuracy))


class VanillaNN(BaseNN):
    prefix = 'vanilla_'

    def train(self, train_iterator, validation_iterator, head, batch_size, epochs, priors=None):
        trainer = Trainer(self.net.collect_params(), self.optimizer, dict(learning_rate=self.learning_rate))

        num_examples = 0
        for epoch in range(epochs):
            cumulative_loss = 0
            for i, batch in enumerate(train_iterator):
                with mx.autograd.record():
                    if self.single_head:
                        output = self.forward(batch.data[0].as_in_context(self.ctx))
                    else:
                        output = self.forward(batch.data[0].as_in_context(self.ctx))[head]
                    labels = batch.label[0].as_in_context(self.ctx)
                    loss = self.loss(output, labels)
                loss.backward()
                trainer.step(batch_size=batch_size, ignore_stale_grad=True)
                cumulative_loss += mx.nd.sum(loss).asscalar()
                num_examples += len(labels)

            train_iterator.reset()
            validation_iterator.reset()
            train_accuracy = self.evaluate_accuracy(train_iterator, head=head)
            validation_accuracy = self.evaluate_accuracy(validation_iterator, head=head)
            self.print_status(epoch + 1, cumulative_loss / num_examples, train_accuracy, validation_accuracy)


class BayesianNN(BaseNN):
    prefix = 'bayesian_'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_model()

    def create_model(self):
        self.model = Model(verbose=self.verbose)
        self.model.N = Variable()
        self.model.f = MXFusionGluonFunction(self.net, num_outputs=self.num_heads, broadcastable=False)
        self.model.x = Variable(shape=(self.model.N, self.network_shape[0]))

        if self.single_head:
            self.model.r = self.model.f(self.model.x)
            self.model.y = Categorical.define_variable(
                log_prob=self.model.r, shape=(self.model.N, 1), num_classes=self.network_shape[-1])
            self.create_prior_variables(self.model.r)
        else:
            r = self.model.f(self.model.x)
            for head, label_shape in enumerate(self.network_shape[-1]):
                rh = r[head] if self.num_heads > 1 else r
                setattr(self.model, 'r{}'.format(head), rh)
                y = Categorical.define_variable(log_prob=rh, shape=(self.model.N, 1), num_classes=label_shape)
                setattr(self.model, 'y{}'.format(head), y)
                # TODO the statement below could probably be done only for the first head, since they all share the same
                # factor parameters
                self.create_prior_variables(rh)

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
            r = getattr(self.model, 'r{}'.format(head))
        return r.factor.parameters

    # noinspection PyUnresolvedReferences
    def train(self, train_iterator, validation_iterator, head, batch_size, epochs, priors=None):
        for i, batch in enumerate(train_iterator):
            if i > 0:
                raise NotImplementedError("Currently not supported for more than one batch of data. "
                                          "Please switch to using the MinibatchInferenceLoop")

            data = mx.nd.flatten(batch.data[0]).as_in_context(self.ctx)
            labels = mx.nd.expand_dims(batch.label[0], axis=-1).as_in_context(self.ctx)

            if self.verbose:
                print("Data shape {}".format(data.shape))

            # pass some data to initialise the net
            self.net(data[:1])

            if self.single_head:
                observed = [self.model.x, self.model.y]
                ignored = None
                kwargs = dict(y=labels, x=data)
            else:
                observed = [self.model.x, getattr(self.model, "y{}".format(head))]
                y_other = [getattr(self.model, "y{}".format(h)) for h in range(self.num_heads) if h != head]
                r_other = [getattr(self.model, "r{}".format(h)) for h in range(self.num_heads) if h != head]
                ignored = y_other + r_other
                kwargs = {'x': data, 'y{}'.format(head): labels, 'ignored': dict((v.name, v) for v in ignored)}

            q = create_Gaussian_meanfield(model=self.model, ignored=ignored, observed=observed)
            alg = StochasticVariationalInference(num_samples=5, model=self.model, posterior=q, observed=observed,
                                                 ignored=ignored)
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

            if self.single_head:
                print("Running single-headed inference")
            else:
                print("Running multi-headed inference for head {}".format(head))
            self.inference.run(max_iter=self.max_iter, learning_rate=self.learning_rate,
                               verbose=False, callback=self.print_status, **kwargs)

    # noinspection PyUnresolvedReferences
    @property
    def posteriors(self):
        q = self.inference.inference_algorithm.posterior

        # TODO: don't convert to numpy arrays
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
                    # print("Head {}, variable {}, shape {}"
                    #       .format(head, v.inherited_name, posteriors[v.inherited_name + '_mean'].shape))
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

            # pass some data to initialise the net
            self.net(data[:1])

            r = self.model.r if self.single_head else getattr(self.model, 'r{}'.format(head))
            y = self.model.y if self.single_head else getattr(self.model, 'y{}'.format(head))
            y_other = [getattr(self.model, "y{}".format(h)) for h in range(self.num_heads) if h != head]
            r_other = [getattr(self.model, "r{}".format(h)) for h in range(self.num_heads) if h != head]
            ignored = y_other + r_other

            if self.verbose:
                print("Data shape {}".format(data.shape))

            if len(ignored) > 0:
                # Here we need to re-instantiate the ignored variables into the posterior if they don't exist
                model = self.inference.inference_algorithm.model  # .clone()
                old_posterior = self.inference.inference_algorithm.posterior
                new_posterior = old_posterior.clone(model=model)

                # Reattach the missing parts of the graph
                new_posterior[r].set_prior(new_posterior[r.factor])
                new_posterior[r].factor.predecessors = [(k, new_posterior[v]) for k, v in r.factor.predecessors]
                new_posterior[r].factor.successors = [(k, new_posterior[v]) for k, v in r.factor.successors]
                new_posterior[y].set_prior(new_posterior[y.factor])
                new_posterior[y].factor.predecessors = [(k, new_posterior[v]) for k, v in y.factor.predecessors]
                new_posterior[y].factor.successors = [(k, new_posterior[v]) for k, v in y.factor.successors]

                # Set the posterior to be the new posterior
                self.inference.inference_algorithm._extra_graphs[0] = new_posterior
            else:
                old_posterior = None

            prediction_inference = VariationalPosteriorForwardSampling(
                num_samples=10, observed=[self.model.x],
                inherited_inference=self.inference,
                target_variables=[r],
                ignored=ignored  # dict((v.name, v) for v in ignored)
            )

            res = prediction_inference.run(x=mx.nd.array(data))

            if old_posterior is not None:
                # Set the posterior back to the old posterior
                self.inference.inference_algorithm._extra_graphs[0] = old_posterior

            return res[0].asnumpy()

    @staticmethod
    def prior_mean(shape):
        return mx.nd.zeros(shape=shape)

    @staticmethod
    def prior_variance(shape):
        return mx.nd.ones(shape=shape) * 3
