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


from .dist_batch_loop import DistributedBatchInferenceLoop
from .inference import Inference
from .dist_minibatch_loop import DistributedMinibatchInferenceLoop
from ..util.inference import discover_shape_constants, init_outcomes
from .dist_grad_loop import DistributedGradLoop

class DistributedGradBasedInference(Inference):
    """
    The abstract class for distributed gradient-based inference methods.
    An inference method consists of a few components: the applied inference algorithm, the model definition
    (optionally a definition of posterior approximation), the inference parameters.

    :param inference_algorithm: The applied inference algorithm
    :type inference_algorithm: InferenceAlgorithm
    :param grad_loop: The reference to the main loop of gradient optimization
    :type grad_loop: GradLoop
    :param constants: Specify a list of model variables as constants
    :type constants: {Variable: mxnet.ndarray}
    :param hybridize: Whether to hybridize the MXNet Gluon block of the inference method.
    :type hybridize: boolean
    :param dtype: data type for internal numerical representation
    :type dtype: {numpy.float64, numpy.float32, 'float64', 'float32'}
    :param context: The MXNet context
    :type context: {mxnet.cpu or mxnet.gpu}
    :param logger: The logger to send logs to
    :type logger: :class:`inference.Logger`
    :param rv_scaling: the scaling factor of random variables
    :type rv_scaling: {Variable: scaling factor}
    """

    def __init__(self, inference_algorithm, grad_loop=None, constants=None,
                 hybridize=False, dtype=None, context=None, logger=None, rv_scaling=None):
        if grad_loop is None:
            grad_loop = DistributedBatchInferenceLoop()
        if not (isinstance(grad_loop, DistributedGradLoop)):
            raise TypeError("grad_loop must be a type of DistributedGradLoop.")

        super(DistributedGradBasedInference, self).__init__(
            inference_algorithm=inference_algorithm, constants=constants,
            hybridize=hybridize, dtype=dtype, context=context, logger=logger)
        self._grad_loop = grad_loop
        self.rv_scaling = rv_scaling

    def rescale(self, rv_scaling):
        """
        Return the rescaled scaling factor of random variables for SVI.
        """
        import horovod.mxnet as hvd
        for _, variable in enumerate(
                self.inference_algorithm.model.get_latent_variables(self.inference_algorithm.observed)):
            if variable not in rv_scaling:
                rv_scaling[variable.uuid] = 1 / hvd.size()

        return rv_scaling

    def create_executor(self):
        """
        Return a MXNet Gluon block responsible for the execution of the inference method.
        """
        from mxfusion.inference import StochasticVariationalInference
        rv_scaling = self.scale_if_minibatch(self._grad_loop)
        if isinstance(self.inference_algorithm, StochasticVariationalInference):
            rv_scaling = self.rescale(rv_scaling=rv_scaling)

        infr = self._inference_algorithm.create_executor(
            data_def=self.observed_variable_UUIDs, params=self.params,
            var_ties=self.params.var_ties, rv_scaling=rv_scaling)
        if self._hybridize:
            infr.hybridize()
        infr.initialize(ctx=self.mxnet_context)
        return infr

    def run(self, optimizer='adam', learning_rate=1e-3, max_iter=2000,
            verbose=False, **kwargs):
        """
        Run the inference method.

        :param optimizer: the choice of optimizer (default: 'adam')
        :type optimizer: str
        :param learning_rate: the learning rate of the gradient optimizer (default: 0.001)
        :type learning_rate: float
        :param max_iter: the maximum number of iterations of gradient optimization
        :type max_iter: int
        :param verbose: whether to print per-iteration messages.
        :type verbose: boolean
        :param kwargs: The keyword arguments specify the data for inferences. The key of each argument is the name of
        the corresponding variable in model definition and the value of the argument is the data in numpy array format.
        """
        data = [kwargs[v] for v in self.observed_variable_names]
        self.initialize(**kwargs)

        infr = self.create_executor()

        if isinstance(self._grad_loop, DistributedMinibatchInferenceLoop):
            def update_shape_constants(data_batch):
                data_shapes = {i: d.shape for i, d in zip(self.observed_variable_UUIDs,
                                                          data_batch)}
                shape_constants = discover_shape_constants(data_shapes, self._graphs)
                self.params.update_constants(shape_constants)

            return self._grad_loop.run(
                infr_executor=infr, data=data, param_dict=self.params.param_dict,
                ctx=self.mxnet_context, optimizer=optimizer,
                learning_rate=learning_rate, max_iter=max_iter,
                update_shape_constants=update_shape_constants, verbose=verbose, logger=self._logger)
        else:
            return self._grad_loop.run(
                infr_executor=infr, data=data, param_dict=self.params.param_dict,
                ctx=self.mxnet_context, optimizer=optimizer,
                learning_rate=learning_rate, max_iter=max_iter, verbose=verbose, logger=self._logger)


class GradTransferInference(DistributedGradBasedInference):
    """
    The abstract Inference method for transferring the outcome of one inference
    method to another.

    :param inference_algorithm: The applied inference algorithm
    :type inference_algorithm: InferenceAlgorithm
    :param train_params:
    :param constants: Specify a list of model variables as constants
    :type constants: {Variable: mxnet.ndarray}
    :param hybridize: Whether to hybridize the MXNet Gluon block of the inference method.
    :type hybridize: boolean
    :param dtype: data type for internal numerical representation
    :type dtype: {numpy.float64, numpy.float32, 'float64', 'float32'}
    :param context: The MXNet context
    :type context: {mxnet.cpu or mxnet.gpu}
    """

    def __init__(self, inference_algorithm, infr_params, train_params,
                 grad_loop=None, var_tie=None,
                 constants=None, hybridize=False,
                 dtype=None, context=None):
        self._var_tie = var_tie if var_tie is not None else {}
        self._inherited_params = infr_params
        self.train_params = train_params
        super(GradTransferInference, self).__init__(
            inference_algorithm=inference_algorithm,
            grad_loop=grad_loop, constants=constants,
            hybridize=hybridize, dtype=dtype, context=context)

    def _initialize_params(self):
        self.params.initialize_with_carryover_params(
            self._graphs, self.observed_variable_UUIDs, self._var_tie,
            init_outcomes(self._inherited_params))
        self.params.fix_all()
