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


import warnings
import numpy as np
import mxnet as mx
from .inference_parameters import InferenceParameters
from ..common.config import get_default_device, get_default_dtype
from ..common.exceptions import InferenceError
from ..util.inference import discover_shape_constants, init_outcomes
from ..models import FactorGraph, Model, Posterior


class Inference(object):
    """
    Abstract class defining an inference method that can be applied to a model.
    An inference method consists of a few components: the applied inference algorithm,
    the model definition (optionally a definition of posterior
    approximation), and the inference parameters.

    :param inference_algorithm: The applied inference algorithm
    :type inference_algorithm: InferenceAlgorithm
    :param constants: Specify a list of model variables as constants
    :type constants: {Variable: mxnet.ndarray}
    :param hybridize: Whether to hybridize the MXNet Gluon block of the inference method.
    :type hybridize: boolean
    :param dtype: data type for internal numerical representation
    :type dtype: {numpy.float64, numpy.float32, 'float64', 'float32'}
    :param context: The MXNet context
    :type context: {mxnet.cpu or mxnet.gpu}
    """

    def __init__(self, inference_algorithm, constants=None,
                 hybridize=False, dtype=None, context=None):

        self.dtype = dtype if dtype is not None else get_default_dtype()
        self.mxnet_context = context if context is not None else get_default_device()
        self._hybridize = hybridize
        self._graphs = inference_algorithm.graphs
        self._inference_algorithm = inference_algorithm
        self.params = InferenceParameters(constants=constants,
                                          dtype=self.dtype,
                                          context=self.mxnet_context)
        self._initialized = False

    def print_params(self):
        """
        Returns a string with the inference parameters nicely formatted for display, showing which model they came from and their name + uuid.
        
        Format:
        > infr.print_params()
        Variable(1ab23)(name=y) - (Model/Posterior(123ge2)) - (first mxnet values/shape)
        """
        def get_class_name(graph):
            if isinstance(graph, Model):
                return "Model"
            elif isinstance(graph, Posterior):
                return "Posterior"
            else:
                return "FactorGraph"
        string = ""
        for param_uuid, param in self.params.param_dict.items():
            temp = [(graph,graph[param_uuid]) for i,graph in enumerate(self._graphs) if param_uuid in graph]
            graph, var_param = temp[0]
            string += "{} in {}({}) : {} \n\n".format(var_param, get_class_name(graph), graph._uuid[:5], param.data())
        return string

    @property
    def observed_variables(self):
        return self._inference_algorithm.observed_variables

    @property
    def observed_variable_UUIDs(self):
        return self._inference_algorithm.observed_variable_UUIDs

    @property
    def observed_variable_names(self):
        return self._inference_algorithm.observed_variable_names

    @property
    def graphs(self):
        """
        Returns the list of graphs the inference algorithm operates over.
        """
        return self._graphs

    @property
    def inference_algorithm(self):
        """
        Return the reference to the used inference algorithm.
        """
        return self._inference_algorithm

    def create_executor(self):
        """
        Return a MXNet Gluon block responsible for the execution of the inference method.
        """
        infr = self._inference_algorithm.create_executor(data_def=self.observed_variable_UUIDs,
                                                         params=self.params,
                                                         var_ties=self.params.var_ties)
        if self._hybridize:
            infr.hybridize()
        infr.initialize(ctx=self.mxnet_context)
        return infr

    def _initialize_params(self):
        self.params.initialize_params(self._graphs, self.observed_variable_UUIDs)

    def initialize(self, **kw):
        """
        Initialize the inference method with the shapes of observed variables.
        The inputs of the keyword arguments are the names of the
        variables in the model definition. The values of the keyword arguments are the data of the
        corresponding variables (mxnet.ndarray)
        or their shape (tuples).
        """
        if not self._initialized:
            data = [kw[v] for v in self.observed_variable_names]
            if len(data) > 0:
                if not all(isinstance(d, type(d)) for d in data):
                    raise InferenceError("All items in the keywords must be of the same type. "
                                         "Either all shapes or all data objects.")

                if isinstance(data[0], (tuple, list)):
                    data_shapes = {i: d for i, d in zip(self.observed_variable_UUIDs, data)}
                elif isinstance(data[0], mx.nd.ndarray.NDArray):
                    data_shapes = {i: d.shape for i, d in zip(self.observed_variable_UUIDs,
                                                              data)}
                else:
                    raise InferenceError("Keywords not of type mx.nd.NDArray or tuple/list "
                                         "for shapes passed into initialization.")
                shape_constants = discover_shape_constants(data_shapes,
                                                           self._graphs)
                self.params.update_constants(shape_constants)
            self._initialize_params()
            self._initialized = True
        else:
            warnings.warn("Trying to initialize the inference twice, skipping.")
            # TODO: how to handle when initialization called twice

    def run(self, **kwargs):
        """
        Run the inference method.

        :param **kwargs: The keyword arguments specify the data for inference self. The key of each argument is the name of the corresponding
            variable in model definition and the value of the argument is the data in numpy array format.
        :returns: the samples of target variables (if not specified, the samples of all the latent variables)
        :rtype: {UUID: samples}
        """
        data = [kwargs[v] for v in self.observed_variable_names]
        self.initialize(**kwargs)
        executor = self.create_executor()
        return executor(mx.nd.zeros(1, ctx=self.mxnet_context), *data)

    def set_initializer(self):
        """
        Configure the inference method on how to initialize variables and parameters.
        """
        pass

    def load(self,
             graphs_file=None,
             inference_configuration_file=None,
             parameters_file=None,
             mxnet_constants_file=None,
             variable_constants_file=None):
        """
        Loads back everything needed to rerun an inference algorithm.
        The major pieces of this are the InferenceParameters, FactorGraphs, and
        InferenceConfiguration.

        :param graphs_file: The file containing the graphs to load back for this inference algorithm. The first of these is the primary graph.
        :type graphs_file: str of filename
        :param inference_configuration_file: The file containing any inference specific configuration needed to
        reload this inference algorithm.
            e.g. observation patterns used to train it.
        :type inference_configuration_file: str of filename
        :param parameters_file: These are the parameters of the previous inference algorithm.
        These are in a {uuid: mx.nd.array} mapping.
        :type mxnet_constants_file: file saved down with mx.nd.save(), so a {uuid: mx.nd.array} mapping saved
        in a binary format.
        :param mxnet_constants_file: These are the constants in mxnet format from the previous inference algorithm.
        These are in a {uuid: mx.nd.array} mapping.
        :type mxnet_constants_file: file saved down with mx.nd.save(), so a {uuid: mx.nd.array} mapping saved
        in a binary format.
        :param variable_constants_file: These are the constants in primitive format from the previous
        inference algorithm.
        :type variable_constants_file: json dict of {uuid: constant_primitive}
        """
        graphs = FactorGraph.load_graphs(graphs_file)
        primary_model = graphs[0]
        secondary_graphs = graphs[1:]

        # { current_model_uuid : loaded_uuid}
        self._uuid_map = FactorGraph.reconcile_graphs(
            current_graphs=self.graphs,
            primary_previous_graph=primary_model,
            secondary_previous_graphs=secondary_graphs)
        new_parameters = InferenceParameters.load_parameters(
            uuid_map=self._uuid_map,
            parameters_file=parameters_file,
            variable_constants_file=variable_constants_file,
            mxnet_constants_file=mxnet_constants_file,
            current_params=self.params._params)
        self.params = new_parameters
        self.load_configuration(inference_configuration_file, self._uuid_map)

    def load_configuration(self, config_file, uuid_map):
        """
        Loads relevant inference configuration back from a file.
        Currently only loads the observed variables UUIDs back in, using the uuid_map
        parameter to store the correct current observed variables.

        :param config_file: The file to save the configuration down into.
        :type config_file: str
        :param uuid_map: A map of previous/loaded model component uuids to their current variable in the loaded graph.
        :type uuid_map: { current_model_uuid : loaded_previous_uuid}
        """
        import json
        with open(config_file) as f:
            configuration = json.load(f)
        # loaded_uuid = [uuid_map[uuid] for uuid in configuration['observed']]

    def save_configuration(self, config_file):
        """
        Saves relevant inference configuration down into a file.
        Currently only saves the observed variables UUIDs as {'observed': [observed_uuids]}.

        :param config_file: The file to save the configuration down into.
        :type config_file: str
        """
        import json
        with open(config_file, 'w') as f:
            json.dump({'observed': self.observed_variable_UUIDs}, f, ensure_ascii=False)

    def save(self, prefix=None):
        """
        Saves down everything needed to reload an inference algorithm.
        The two primary pieces of this are the InferenceParameters and FactorGraphs.

        :param prefix: The directory and any appending tag for the files to save this Inference as.
        :type prefix: str , ex. "../saved_inferences/experiment_1"
        """
        prefix = prefix if prefix is not None else "inference"
        self.params.save(prefix=prefix)
        self.save_configuration(prefix + '_configuration.json')
        graphs = [g.as_json()for g in self._graphs]
        FactorGraph.save(prefix + "_graphs.json", graphs)


class TransferInference(Inference):
    """
    The abstract Inference method for transferring the outcome of one inference
    method to another.

    :param inference_algorithm: The applied inference algorithm
    :type inference_algorithm: InferenceAlgorithm
    :param constants: Specify a list of model variables as constants
    :type constants: {Variable: mxnet.ndarray}
    :param hybridize: Whether to hybridize the MXNet Gluon block of the inference method.
    :type hybridize: boolean
    :param dtype: data type for internal numerical representation
    :type dtype: {numpy.float64, numpy.float32, 'float64', 'float32'}
    :param context: The MXNet context
    :type context: {mxnet.cpu or mxnet.gpu}
    """
    def __init__(self, inference_algorithm, infr_params, var_tie=None,
                 constants=None, hybridize=False, dtype=None, context=None):
        self._var_tie = var_tie if var_tie is not None else {}
        self._inherited_params = infr_params
        super(TransferInference, self).__init__(
            inference_algorithm=inference_algorithm, constants=constants,
            hybridize=hybridize, dtype=dtype, context=context)

    def generate_executor(self, **kw):

        data_shapes = [kw[v] for v in self.observed_variable_names]
        if not self._initialized:
            self._initialize_run(self._var_tie, self._inherited_params,
                                 data_shapes)
            self._initialized = True

        infr = self._inference_algorithm.create_executor(
            data_def=self.observed_variable_UUIDs, params=self.params,
            var_ties=self.params.var_ties)
        if self._hybridize:
            infr.hybridize()
        infr.initialize()
        return infr

    def _initialize_params(self):
        self.params.initialize_with_carryover_params(
            self._graphs, self.observed_variable_UUIDs, self._var_tie,
            init_outcomes(self._inherited_params))
