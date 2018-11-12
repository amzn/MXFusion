import mxnet as mx
import numpy as np
from abc import ABC


class BaseNN(ABC):
    def __init__(self, network_shape):
        # input and output placeholders
        self.task_idx = mx.nd.array(name='task_idx', dtype=np.float32)
        self.model = None
        self.network_shape = network_shape
        self.loss = None

    def train(self, train_iter, val_iter, batch_size, ctx):
        #         data = mx.sym.var('data')
        # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
        #         data = mx.sym.flatten(data=data)

        # # create a trainable module on compute context
        # self.model = mx.mod.Module(symbol=self.net, context=ctx)
        self.model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label, ctx=ctx)
        init = mx.init.Xavier(factor_type="in", magnitude=2.34)
        self.model.init_params(initializer=init, force_init=True)
        self.model.fit(train_iter,  # train data
                       eval_data=val_iter,  # validation data
                       optimizer='adam',  # use SGD to train
                       optimizer_params={'learning_rate': 0.001},  # use fixed learning rate
                       eval_metric='acc',  # report accuracy during training
                       batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                       # output progress for each 100 data batches
                       num_epoch=10)  # train for at most 50 dataset passes
        # predict accuracy of mlp
        acc = mx.metric.Accuracy()
        self.model.score(val_iter, acc)
        return acc

    def prediction_prob(self, test_iter, task_idx):
        # TODO task_idx??
        prob = self.model.predict(test_iter)
        return prob


class VanillaNN(BaseNN):
    def __init__(self, network_shape, previous_weights=None):
        super(VanillaNN, self).__init__(network_shape)

        # Create net
        self.net = mx.gluon.nn.HybridSequential(prefix='vanilla_')
        with self.net.name_scope():
            for layer in network_shape[1:-1]:
                self.net.add(mx.gluon.nn.Dense(layer, activation="relu"))
            #  Last layer for classification
            self.net.add(mx.gluon.nn.Dense(network_shape[-1], flatten=True, in_units=network_shape[-2]))
        self.loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()


class MeanFieldNN(BaseNN):
    def __init__(self, network_shape, prior_means, prior_log_variances):
        super(MeanFieldNN, self).__init__(network_shape)
        raise NotImplementedError
