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
from scipy.optimize import fmin_l_bfgs_b

from .grad_loop import GradLoop


class BatchInferenceLoop(GradLoop):
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

        trainer = mx.gluon.Trainer(param_dict,
                                   optimizer=optimizer,
                                   optimizer_params={'learning_rate': learning_rate})
        iter_step = max(max_iter // n_prints, 1)
        for i in range(1, max_iter + 1):
            with mx.autograd.record():
                loss, loss_for_gradient = infr_executor(mx.nd.zeros(1, ctx=ctx), *data)
                loss_for_gradient.backward()

            if logger:
                logger.log("loss", loss.asscalar(), i, newline=not i % iter_step, verbose=verbose)

            trainer.step(batch_size=1, ignore_stale_grad=True)
        infr_executor(mx.nd.zeros(1, ctx=ctx), *data)

        if logger:
            logger.close()


class BatchInferenceLoopScipy(GradLoop):
    """
    The class for the main loop for batch gradient-based optimization.
    """

    def __init__(self):
        super().__init__()
        self._initialized = None

    def run(self, infr_executor, data, param_dict, ctx, optimizer='adam', learning_rate=None, max_iter=1000,
            verbose=False, logger=None):
        """
        :param learning_rate: Not used
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
        :param max_iter: the maximum number of iterations of gradient optimization
        :type max_iter: int
        :param verbose: whether to print per-iteration messages.
        :type verbose: boolean
        :param logger: The logger to send logs to
        :type logger: :class:`inference.Logger`
        """

        if self._initialized is None:
            self._initialize(param_dict)

        x0 = self._extract_array_from_param_dict(param_dict)
        func = lambda x: self._f_df(x, infr_executor, data, ctx, param_dict)

        x_opt, f_opt, _ = fmin_l_bfgs_b(func, x0=x0, maxiter=max_iter)
        print('Final loss:', f_opt)

    def _f_df(self, x, infr_executor, data, ctx, param_dict):
        """
        Objective function for optimization. Returns function value and gradient
        """
        self._set_param_values_from_array(x, param_dict)
        with mx.autograd.record():
            loss, loss_for_gradient = infr_executor(mx.nd.zeros(1, ctx=ctx), *data)
            loss_for_gradient.backward()
        return loss.asnumpy().astype('float64'), self._extract_grad(param_dict).astype('float64')

    def _extract_grad(self, param_dict):
        """
        Extracts gradient information from parameter dictionary
        """
        x = mx.nd.zeros(self._i_end[-1])
        for i, (key, val) in enumerate(param_dict.items()):
            x[self._i_start[i]:self._i_end[i]] = val.data().grad.reshape((-1,))
        return x.asnumpy()

    def _initialize(self, param_dict):
        """
        Stores stuff like shapes of all the parameters so we can flatten them all to one array for scipy and reshape
        them back to the correct shape
        """
        self._i_start = []
        self._i_end = []
        self._shapes = []

        i = 0
        for key, val in param_dict.items():
            self._i_start.append(i)
            self._i_end.append(i + val.data().size)

            i += val.data().size
            self._shapes.append(val.data().shape)

    def _set_param_values_from_array(self, x, param_dict):
        """
        Sets the updated x values in the parameter dictionary
        """
        for i, (key, val) in enumerate(param_dict.items()):
            value_to_set_to = mx.nd.reshape(mx.nd.array(x[self._i_start[i]:self._i_end[i]]), self._shapes[i])
            val.set_data(value_to_set_to)

    def _extract_array_from_param_dict(self, param_dict):
        """
        Extracts the flattened parameter array from the parameter dictionary
        """
        x = mx.nd.zeros(self._i_end[-1])
        for i, (key, val) in enumerate(param_dict.items()):
            x[self._i_start[i]:self._i_end[i]] = val.data().reshape((-1,))
        return x.asnumpy()
