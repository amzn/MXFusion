import mxnet as mx
from .grad_loop import GradLoop


class BatchInferenceLoop(GradLoop):
    """
    The class for the main loop for batch gradient-based optimization.
    """

    def run(self, infr_executor, data, param_dict, ctx, optimizer='adam',
            learning_rate=1e-3, max_iter=2000, n_prints=10, verbose=False):
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
        :param verbose: whether to print per-iteration messages.
        :type verbose: boolean
        """
        trainer = mx.gluon.Trainer(param_dict,
                                   optimizer=optimizer,
                                   optimizer_params={'learning_rate':
                                                     learning_rate})
        iter_step = max_iter // n_prints
        for i in range(max_iter):
            with mx.autograd.record():
                loss, loss_for_gradient = infr_executor(mx.nd.zeros(1, ctx=ctx), *data)
                loss_for_gradient.backward()
            if verbose:
                print('\rIteration {} loss: {}'.format(i + 1, loss.asscalar()),
                      end='')
                if i % iter_step == 0 and i > 0:
                    print()
            trainer.step(batch_size=1, ignore_stale_grad=True)
        loss = infr_executor(mx.nd.zeros(1, ctx=ctx), *data)
