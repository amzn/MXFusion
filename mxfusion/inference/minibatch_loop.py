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
from mxnet.gluon.data import ArrayDataset

from .grad_loop import GradLoop


class MinibatchInferenceLoop(GradLoop):
    """
    The class for the main loop for minibatch gradient-based optimization. The
    *batch_size* specifies the size of mini-batch is used in mini-batch
    inference. *rv_scaling* is used to re-scale the log-likelihood over data
    due to the sub-sampling of the training set. The scaling should be applied
    to all the observed and hidden random variables that are impacted by data
    sub-sampling. The scaling factor is the size of training set dividing the
    batch size.

    :param batch_size: the size of minibatch for optimization
    :type batch_size: int
    :param rv_scaling: the scaling factor of random variables
    :type rv_scaling: {Variable: scaling factor}
    """

    def __init__(self, batch_size=100, rv_scaling=None):
        super(MinibatchInferenceLoop, self).__init__()
        self.batch_size = batch_size
        self.rv_scaling = {v.uuid: s for v, s in rv_scaling.items()} \
            if rv_scaling is not None else rv_scaling

    def run(self, infr_executor, data, param_dict, ctx, optimizer='adam',
            learning_rate=1e-3, max_iter=1000, update_shape_constants=None, logger=None):
        """
        :param infr_executor: The MXNet function that computes the training objective.
        :type infr_executor: MXNet Gluon Block
        :param data: a list of observed variables
        :type data: [mxnet.ndarray]
        :param param_dict: The MXNet ParameterDict for Gradient-based optimization
        :type param_dict: mxnet.gluon.ParameterDict
        :param ctx: MXNet context
        :type ctx: mxnet.cpu or mxnet.gpu
        :param optimizer: the choice of optimizer (default: 'adam')
        :type optimizer: str
        :param learning_rate: the learning rate of the gradient optimizer (default: 0.001)
        :type learning_rate: float
        :param max_iter: the maximum number of iterations of gradient optimization
        :type max_iter: int
        :param logger: The logger to send logs to
        :type logger: :class:`inference.Logger`
        :param update_shape_constants: The callback function to update the shape constants according to the size of minibatch
        :type update_shape_constants: Python function
        """

        if isinstance(data, mx.gluon.data.DataLoader):
            data_loader = data
        else:
            data_loader = mx.gluon.data.DataLoader(
                ArrayDataset(*data), batch_size=self.batch_size, shuffle=True,
                last_batch='rollover')
        trainer = mx.gluon.Trainer(param_dict, optimizer=optimizer, optimizer_params={'learning_rate': learning_rate})

        total_batches = 0
        for e in range(max_iter):
            epoch_loss = batch_number = 0

            for batch_number, data_batch in enumerate(data_loader):
                if not isinstance(data_batch, (list, tuple)):
                    data_batch = [data_batch]

                if update_shape_constants is not None:
                    update_shape_constants(data_batch)

                with mx.autograd.record():
                    loss, loss_for_gradient = infr_executor(mx.nd.zeros(1, ctx=ctx), *data_batch)
                    loss_for_gradient.backward()

                if logger:
                    logger.log(tag='loss', value=loss.asscalar(), step=batch_number + total_batches + 1)

                trainer.step(batch_size=self.batch_size, ignore_stale_grad=True)
                epoch_loss += loss.asscalar()

            if logger:
                logger.log(tag='epoch_loss', value=epoch_loss / batch_number,
                           step=e + 1, iterate_name="Epoch", newline=True)
            total_batches += batch_number
