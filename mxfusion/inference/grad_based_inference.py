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


from .inference import Inference
from .batch_loop import BatchInferenceLoop
from ..util.inference import discover_shape_constants, init_outcomes
from .minibatch_loop import MinibatchInferenceLoop


class GradBasedInference(Inference):
    """
    The abstract class for gradient-based inference methods.
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
    """
    def __init__(self, inference_algorithm, grad_loop=None, constants=None,
                 hybridize=False, dtype=None, context=None):
        if grad_loop is None:
            grad_loop = BatchInferenceLoop()
        super(GradBasedInference, self).__init__(
            inference_algorithm=inference_algorithm, constants=constants,
            hybridize=hybridize, dtype=dtype, context=context)
        self._grad_loop = grad_loop

    def create_executor(self):
        """
        Return a MXNet Gluon block responsible for the execution of the inference method.
        """
        from .minibatch_loop import MinibatchInferenceLoop
        if isinstance(self._grad_loop, MinibatchInferenceLoop):
            rv_scaling = self._grad_loop.rv_scaling
        else:
            rv_scaling = None
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

        if isinstance(self._grad_loop, MinibatchInferenceLoop):
            def update_shape_constants(data_batch):
                data_shapes = {i: d.shape for i, d in zip(self.observed_variable_UUIDs,
                                                          data_batch)}
                shape_constants = discover_shape_constants(data_shapes, self._graphs)
                self.params.update_constants(shape_constants)

            return self._grad_loop.run(
                infr_executor=infr, data=data, param_dict=self.params.param_dict,
                ctx=self.mxnet_context, optimizer=optimizer,
                learning_rate=learning_rate, max_iter=max_iter, verbose=verbose,
                update_shape_constants=update_shape_constants)
        else:
            return self._grad_loop.run(
                infr_executor=infr, data=data, param_dict=self.params.param_dict,
                ctx=self.mxnet_context, optimizer=optimizer,
                learning_rate=learning_rate, max_iter=max_iter, verbose=verbose)

class GradTransferInference(GradBasedInference):
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
