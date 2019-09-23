# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import horovod.mxnet as hvd
from mxnet import gluon, autograd
from .dist_grad_loop import DistributedGradLoop

class DistributedBatchInferenceLoop(DistributedGradLoop):
    """
    The class for the main loop for batch gradient-based optimization.
    """

    def run(self, infr_executor, data, param_dict, ctx, optimizer='adam',
            learning_rate=1e-3, max_iter=1000, n_prints=10, verbose=False, logger=None):
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
        :param n_prints: number of messages to print
        :type n_prints: int
        :param max_iter: the maximum number of iterations of gradient optimization
        :type max_iter: int
        :param verbose: whether to print per-iteration messages.
        :type verbose: boolean
        :param logger: The logger to send logs to
        :type logger: :class:`inference.Logger`
        """
        if logger:
            logger.open()

        trainer = hvd.DistributedTrainer(param_dict, optimizer=optimizer,optimizer_params={'learning_rate': learning_rate})
        data = self.split_data(data=data)

        iter_step = max(max_iter // n_prints, 1)
        for i in range(1, max_iter + 1):
            with autograd.record():
                loss, loss_for_gradient = infr_executor(mx.nd.zeros(1, ctx=ctx), *data)
                # stepping up the learning rate for distributed training
                loss_for_gradient = loss_for_gradient * hvd.size()
            loss_for_gradient.backward()

            if logger:
                logger.log("loss", loss.asscalar(), i, newline=not i % iter_step, verbose=verbose)

            trainer.step(batch_size=1, ignore_stale_grad=True)

        infr_executor(mx.nd.zeros(1, ctx=ctx), *data)

        if logger:
            logger.close()