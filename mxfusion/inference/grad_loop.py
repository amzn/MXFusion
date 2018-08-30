from abc import ABC, abstractmethod


class GradLoop(ABC):
    """
    The abstract class for the loop of gradient-based inference.
    """

    @abstractmethod
    def run(self, infr_executor, data, param_dict, ctx, optimizer='adam',
            learning_rate=1e-3, max_iter=2000, verbose=False):
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
        pass
