# ==============================================================================
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


# Bayesian Neural Network (VI) for classification benchmarking

import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon.data import DataLoader
from mxnet.gluon import Trainer
import numpy as np
from mxnet.gluon.data.vision import MNIST, FashionMNIST, CIFAR10, CIFAR100
from mxnet.ndarray import softmax_cross_entropy
import logging
from mxfusion.inference import VariationalPosteriorForwardSampling, create_Gaussian_meanfield, \
    StochasticVariationalInference, GradIteratorBasedInference, MinibatchInferenceLoop
from mxfusion.components.functions.operators import broadcast_to
from mxfusion.components.distributions import Normal, Categorical
from mxfusion import Variable, Model
from mxfusion.components.functions import MXFusionGluonFunction
from tqdm import tqdm
from mlp import MLP
import json

import warnings
from functools import wraps
from time import time

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__!r} took: {te-ts:2.4f} sec')
        return result

    return wrap


# Monkey patch data loader printing
DataLoader.__repr__ = lambda self: f"{self.__class__.__name__}()"


class VanillaNN:
    def __init__(self, architecture, metrics, ctx):
        self.net = MLP(prefix=self.__class__.__name__, network_shape=architecture)
        self.net.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)
        self.metrics = [metric_func() for metric_func in metrics]
        self.validation_scores = dict((metric.name, []) for metric in self.metrics)
        self.ctx = ctx

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_dims}, {self.num_classes}, {self.ctx})"

    @timing
    def train(self, train_loader, val_loader, batch_size, epochs, optimizer, optimizer_params, **kwargs):
        trainer = Trainer(params=self.net.collect_params(), optimizer=optimizer, optimizer_params=optimizer_params)
        cumulative_loss = 0

        for metric in self.metrics:
            metric.reset()
            self.validation_scores = dict((metric.name, []) for metric in self.metrics)

        for e in range(epochs):
            for data, label in tqdm(iter(train_loader)):
                with autograd.record():
                    output = self.net(data.as_in_context(self.ctx))
                    loss = softmax_cross_entropy(output, label.as_in_context(self.ctx))
                    loss.backward()
                trainer.step(data.shape[0])
                cumulative_loss += nd.sum(loss).asscalar()
            self.epoch_callback(e, cumulative_loss, val_loader)

    def epoch_callback(self, e, cumulative_loss, val_loader):
        self.update_metrics(val_loader)

        # TODO: this depends on accuracy being the first metric in the list
        validation_accuracy = self.metrics[0].get()[1]
        print(f"epoch {e + 1}. Loss: {cumulative_loss}, Validation accuracy {validation_accuracy}")

        # Update validation scores
        for metric in self.metrics:
            self.validation_scores[metric.name].append(metric.get()[1])

    def _update_metrics(self, output, label):
        predictions = nd.argmax(output, axis=1)
        probs = nd.softmax(output)
        for metric in self.metrics:
            if metric.name in ('mse', 'nll-loss'):
                preds = probs
            else:
                preds = predictions
            metric.update(preds=preds, labels=label.as_in_context(self.ctx))

    @timing
    def update_metrics(self, data_loader):
        for data, label in iter(data_loader):
            output = self.net(data.as_in_context(self.ctx))
            self._update_metrics(output, label)


class MeanFieldNN(VanillaNN):
    def __init__(self, architecture, metrics, ctx):
        super().__init__(architecture, metrics, ctx)
        m = Model()
        m.N = Variable()
        m.f = MXFusionGluonFunction(self.net, num_outputs=1, broadcastable=False)
        m.x = Variable(shape=(m.N, self.net.network_shape[0]))
        m.r = m.f(m.x)
        for _, v in m.r.factor.parameters.items():
            v.set_prior(Normal(mean=broadcast_to(mx.nd.array([0], ctx=self.ctx), v.shape),
                               variance=broadcast_to(mx.nd.array([1.], ctx=self.ctx), v.shape)))
        m.y = Categorical.define_variable(log_prob=m.r, shape=(m.N, 1), num_classes=self.net.network_shape[-1])
        # print(m)
        self.model = m
        self.inference = None

    @timing
    def train(self, train_loader, val_loader, batch_size, epochs, optimizer, optimizer_params, **kwargs):
        for metric in self.metrics:
            metric.reset()
            self.validation_scores = dict((metric.name, []) for metric in self.metrics)

        data_shape = train_loader._dataset._data.shape

        # Set the initial scaling if not supplied
        initial_scaling = kwargs.get('initial_scaling', 1e-6)

        # Pass the first batch of data through the network to initialise it
        x, y = next(iter(train_loader))
        x = x.as_in_context(self.ctx)
        y = y.as_in_context(self.ctx)
        self.net.forward(x)

        # Setup the inference procedure
        observed = [self.model.x, self.model.y]
        q = create_Gaussian_meanfield(model=self.model, observed=observed)
        alg = StochasticVariationalInference(num_samples=5, model=self.model, posterior=q, observed=observed)
        rv_scaling = data_shape[0] / (batch_size * epochs)
        grad_loop = MinibatchInferenceLoop(batch_size=batch_size, rv_scaling={self.model.y: rv_scaling})
        self.inference = GradIteratorBasedInference(inference_algorithm=alg, grad_loop=grad_loop, context=self.ctx)
        self.inference.initialize(x=x, y=y)

        # Initialise the NN weights
        for v_name, v in self.model.r.factor.parameters.items():
            self.inference.params[q[v].factor.mean] = self.net.collect_params()[v_name].data().as_in_context(ctx)
            self.inference.params[q[v].factor.variance] = mx.nd.ones_like(
                self.inference.params[q[v].factor.variance], ctx=self.ctx) * initial_scaling

        self.inference.run(data=train_loader, max_iter=epochs, **optimizer_params, verbose=True, x=None, y=None,
                           callback=lambda *args, **kwargs: self.epoch_callback(*args, **kwargs, val_loader=val_loader))

    @timing
    def update_metrics(self, data_loader):
        if self.inference is None:
            raise ValueError("Model not yet trained")

        inference = VariationalPosteriorForwardSampling(10, [self.model.x], self.inference, [self.model.r], context=ctx)

        for data, label in iter(data_loader):
            res = inference.run(x=data.as_in_context(self.ctx))
            output = nd.mean(res[0], axis=0)
            self._update_metrics(output, label)


def get_data(data_class, batch_size, ctx):
    from multiprocessing import cpu_count
    cpu_count = 1 if ctx == mx.context.gpu() else cpu_count()
    print(f"CPU count {cpu_count}")

    # Load MNIST Data
    def transform(data, label):
        return data.reshape(-1).astype('float32') / 255, label.astype('float32')

    train_dataset = data_class(train=True, transform=transform)
    valid_dataset = data_class(train=False, transform=transform)

    data_shape = train_dataset._data.shape
    data_shape = data_shape[0], int(np.product(data_shape[1:]))
    num_classes = len(np.unique(train_dataset._label))

    # TODO: Since we're not doing any parameter tuning, the validation and test sets are the same
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=cpu_count)
    valid_data_loader = DataLoader(valid_dataset, batch_size, num_workers=cpu_count)
    test_data_loader = DataLoader(valid_dataset, batch_size, num_workers=cpu_count)

    return train_data_loader, valid_data_loader, test_data_loader, num_classes, data_shape


if __name__ == "__main__":
    # Fix the seed
    mx.random.seed(42)
    np.random.seed(42)

    # Set the compute context, GPU is available otherwise CPU
    ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    print(f"Context: {ctx}")

    batch_size = 100

    # Different numbers of hidden layers
    hidden_choices = (
        (1000,),
        (128, 64),
        (2500, 2000, 1500, 1000, 500)
    )

    # Different models
    models = {
        VanillaNN: dict(epochs=10, optimizer='sgd', optimizer_params=dict(learning_rate=0.05)),
        MeanFieldNN: dict(epochs=10, optimizer='adam', optimizer_params=dict(learning_rate=0.001),
                          initial_scaling=1e-9)
    }

    # Datasets
    datasets = (MNIST, FashionMNIST, CIFAR10, CIFAR100)

    metrics = (
        mx.metric.Accuracy,
        mx.metric.MSE,
        mx.metric.NegativeLogLikelihood
    )

    with open('results.txt', 'w') as f:
        for data_class in datasets:
            train_data_loader, valid_data_loader, test_data_loader, num_classes, data_shape = \
                get_data(data_class, batch_size, ctx)

            for hidden in hidden_choices:
                architecture = (data_shape[1],) + hidden + (num_classes,)

                # for model_class in VanillaNN, MeanFieldNN:
                for model_class, run_args in models.items():
                    print("--------------------------------------")
                    print(f"{model_class.__name__} on {data_class.__name__}")
                    print(f"Data shape: {data_shape}")
                    print(f"Architecture: {architecture}")
                    print(f"Arguments: {run_args}")

                    nn_wrapper = model_class(architecture, metrics=metrics, ctx=ctx)
                    nn_wrapper.train(train_data_loader, valid_data_loader, batch_size, **run_args)
                    nn_wrapper.update_metrics(test_data_loader)

                    evaluations = dict()
                    for metric in nn_wrapper.metrics:
                        evaluations[metric.name] = metric.get()[1]
                        print(f"Final test {metric.name}: {evaluations[metric.name]}")
                    print()

                    results = dict(data=data_class.__name__,
                                   architecture=architecture,
                                   run_args=run_args,
                                   model=model_class.__name__,
                                   evaluations=evaluations,
                                   validation_scores=nn_wrapper.validation_scores)
                    f.write(json.dumps(results) + '\n')
                    f.flush()
