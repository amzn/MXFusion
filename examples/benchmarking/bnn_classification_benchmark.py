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
from matplotlib.pyplot import plot
from mxnet import autograd, nd
from mxnet.gluon.data import DataLoader
from mxnet.gluon import nn, Trainer
import numpy as np
from mxnet.gluon.data.vision import MNIST
from mxnet.ndarray import softmax_cross_entropy
import logging
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.inference import VariationalPosteriorForwardSampling, BatchInferenceLoop, create_Gaussian_meanfield, \
    GradBasedInference, StochasticVariationalInference, GradIteratorBasedInference, MinibatchInferenceLoop
from mxfusion.components.functions.operators import broadcast_to
from mxfusion.components.distributions import Normal, Categorical
from mxfusion import Variable, Model
from mxfusion.components.functions import MXFusionGluonFunction
from tqdm import tqdm

import warnings
from multiprocessing import cpu_count
from functools import wraps
from time import time

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
CPU_COUNT = cpu_count()


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print(f'func:{f.__name__!r} args:[{args!r}, {kw!r}] took: {te-ts:2.4f} sec')
        print(f'func:{f.__name__!r} took: {te-ts:2.4f} sec')
        return result

    return wrap


# Monkey patch modified factor printing
from mxfusion.components import Factor


def print_factor(self):
    out_str = self.__class__.__name__
    if self.predecessors is not None:
        out_str += '(' + ', '.join([str(name) + '=' + str(var) for name, var in self.predecessors]) \
                   + (f", shape={self.outputs[0][1].shape})" if len(self.outputs) == 1 else ')')
    return out_str


Factor.__repr__ = print_factor

# Monkey patch data loader printing
DataLoader.__repr__ = lambda self: f"{self.__class__.__name__}()"


class VanillaNN:
    def __init__(self, num_dims, num_classes, ctx):
        net = nn.HybridSequential(prefix='nn_')
        with net.name_scope():
            net.add(nn.Dense(128, activation="relu", flatten=False, in_units=num_dims))
            net.add(nn.Dense(64, activation="relu", flatten=False, in_units=128))
            net.add(nn.Dense(num_classes, flatten=False, in_units=64))
        net.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)
        net.hybridize(static_alloc=True, static_shape=True)
        self.num_classes = num_classes
        self.num_dims = num_dims
        self.net = net
        self.ctx = ctx

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_dims}, {self.num_classes}, {self.ctx})"

    @timing
    def train(self, train_loader, val_loader, batch_size, epochs, optimizer, optimizer_params):
        trainer = Trainer(params=self.net.collect_params(), optimizer=optimizer, optimizer_params=optimizer_params)
        cumulative_loss = 0

        for e in range(epochs):
            for data, label in tqdm(iter(train_loader)):
                with autograd.record():
                    output = self.net(data.as_in_context(self.ctx))
                    loss = softmax_cross_entropy(output, label.as_in_context(self.ctx))
                    loss.backward()
                trainer.step(data.shape[0])
                cumulative_loss += nd.sum(loss).asscalar()
            self.print_progress(e, cumulative_loss, val_loader)

    def print_progress(self, e, cumulative_loss, val_loader):
        validation_accuracy = self.evaluate_accuracy(val_loader)
        print(f"Epoch {e}. Loss: {cumulative_loss}, Validation accuracy {validation_accuracy}")

    @timing
    def predict(self, data_loader):
        predictions = nd.zeros(shape=(data_loader.num_data,))
        i = 0
        for data in iter(data_loader):
            with autograd.predict_mode():
                output = self.net(data.as_in_context(self.ctx))
                predictions[i:i + data.shape[0]] = nd.argmax(output, axis=1)
                i += data.shape[0]
                i += data.shape[0]
        return predictions

    @timing
    def evaluate_accuracy(self, data_loader):
        acc = mx.metric.Accuracy()
        for data, label in iter(data_loader):
            output = self.net(data.as_in_context(self.ctx))
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label.as_in_context(self.ctx))
        return acc.get()[1]


class MeanFieldNN(VanillaNN):
    def __init__(self, num_dims, num_classes, ctx):
        super().__init__(num_dims, num_classes, ctx)
        m = Model()
        m.N = Variable()
        m.f = MXFusionGluonFunction(self.net, num_outputs=1, broadcastable=False)
        m.x = Variable(shape=(m.N, num_dims))
        m.r = m.f(m.x)
        for _, v in m.r.factor.parameters.items():
            v.set_prior(Normal(mean=broadcast_to(mx.nd.array([0]), v.shape),
                               variance=broadcast_to(mx.nd.array([1.]), v.shape)))
        m.y = Categorical.define_variable(log_prob=m.r, shape=(m.N, 1), num_classes=num_classes)
        print(m)
        self.model = m
        self.inference = None

    @timing
    def train(self, train_loader, val_loader, batch_size, epochs, optimizer, optimizer_params):
        data_shape = train_loader._dataset._data.shape

        # Create some dummy data and pass it through the net to initialise it
        # x_init = nd.random.normal(shape=(batch_size, self.num_dims))
        # y_init = nd.random.multinomial(
        #     data=nd.ones(self.num_classes) / float(self.num_classes), shape=(batch_size, 1))
        x, y = next(iter(train_loader))
        self.net.forward(x.as_in_context(self.ctx))

        observed = [self.model.x, self.model.y]
        q = create_Gaussian_meanfield(model=self.model, observed=observed)
        alg = StochasticVariationalInference(num_samples=5, model=self.model, posterior=q, observed=observed)
        rv_scaling = data_shape[0] / (batch_size * epochs)
        grad_loop = MinibatchInferenceLoop(batch_size=batch_size, rv_scaling={self.model.y: rv_scaling})
        self.inference = GradIteratorBasedInference(inference_algorithm=alg, grad_loop=grad_loop)
        # self.inference = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())

        # self.inference.initialize(x=x_init, y=y_init)
        self.inference.initialize(x=x, y=y)

        for v_name, v in self.model.r.factor.parameters.items():
            self.inference.params[q[v].factor.mean] = self.net.collect_params()[v_name].data()
            self.inference.params[q[v].factor.variance] = mx.nd.ones_like(
                self.inference.params[q[v].factor.variance]) * 1e-6

        self.inference.run(data=train_loader, max_iter=epochs, **optimizer_params, verbose=True, x=None, y=None,
                           callback=lambda *args, **kwargs: self.print_progress(*args, **kwargs, val_loader=val_loader))

    @timing
    def evaluate_accuracy(self, data_loader):
        acc = mx.metric.Accuracy()

        if self.inference is None:
            raise ValueError("Model not yet trained")

        inference = VariationalPosteriorForwardSampling(10, [self.model.x], self.inference, [self.model.r])

        for data, label in iter(data_loader):
            res = inference.run(x=data)
            predictions = nd.argmax(nd.mean(res[0], axis=0), axis=1)
            acc.update(preds=predictions, labels=label.as_in_context(self.ctx))
        return acc.get()[1]


def get_mnist(batch_size):
    # Load MNIST Data
    def transform(data, label):
        return data.reshape(-1).astype('float32') / 255, label.astype('float32')

    train_dataset = MNIST(train=True, transform=transform)
    valid_dataset = MNIST(train=False, transform=transform)
    num_classes = 10

    data_shape = train_dataset._data.shape
    data_shape = data_shape[0], int(np.product(data_shape[1:]))

    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=CPU_COUNT)
    valid_data_loader = DataLoader(valid_dataset, batch_size, num_workers=CPU_COUNT)
    test_data_loader = DataLoader(valid_dataset, batch_size, num_workers=CPU_COUNT)

    return MNIST.__name__, train_data_loader, valid_data_loader, test_data_loader, num_classes, data_shape


if __name__ == "__main__":
    # Fix the seed
    mx.random.seed(42)

    # Set the compute context, GPU is available otherwise CPU
    ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    print(f"Context: {ctx}")

    batch_size = 100
    data_name, train_data_loader, valid_data_loader, test_data_loader, num_classes, data_shape = get_mnist(batch_size)

    models = {
        # VanillaNN: dict(epochs=5, optimizer='sgd', optimizer_params=dict(learning_rate=0.1)),
        MeanFieldNN: dict(epochs=10, optimizer='sgd', optimizer_params=dict(learning_rate=0.1))
    }

    # for model_class in VanillaNN, MeanFieldNN:
    for model_class, run_args in models.items():
        print("--------------------------------------")
        print(f"{model_class.__name__} on {data_name}")
        print(f"Data shape: {data_shape}")

        nn_wrapper = model_class(data_shape[1], num_classes, ctx=ctx)
        nn_wrapper.train(train_data_loader, valid_data_loader, batch_size, **run_args)
        acc = nn_wrapper.evaluate_accuracy(test_data_loader)
        print(f"Final test accuracy: {acc}")
        assert acc > 0.96, f"Achieved accuracy ({acc:f}) is lower than expected (0.96)"
        print()
