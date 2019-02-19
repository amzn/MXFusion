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
import io
import json
import numpy as np
import mxnet as mx
import zipfile
from .inference_parameters import InferenceParameters
from ..common.config import get_default_device, get_default_dtype
from ..common.exceptions import InferenceError
from ..util.inference import discover_shape_constants, init_outcomes
from ..models import FactorGraph, Model, Posterior
from ..util.serialization import ModelComponentEncoder, make_numpy, load_json_from_zip, load_parameters, \
                                 FILENAMES, DEFAULT_ZIP, ENCODINGS, SERIALIZATION_VERSION


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
             zip_filename=DEFAULT_ZIP):
        """
        Loads back everything needed to rerun an inference algorithm from a zip file.
        See the save function for details on the structure of the zip file.

        :param zip_filename: Path to the zip file of the inference method to load back in.
                             Defaults to the default name of inference.zip
        :type zip_filename: str of zip filename
        """

        # Check version is correct
        zip_version = load_json_from_zip(zip_filename, FILENAMES['version_file'])
        if zip_version['serialization_version'] != SERIALIZATION_VERSION:
            raise SerializationError("Serialization version of saved inference \
                                     and running code are note the same.")

        # Load parameters back in

        with zipfile.ZipFile(zip_filename, 'r') as zip_file:
            mxnet_parameters = load_parameters(FILENAMES['mxnet_params'], zip_file, context=self.mxnet_context)
            mxnet_constants = load_parameters(FILENAMES['mxnet_constants'], zip_file, context=self.mxnet_context)

        variable_constants = load_json_from_zip(zip_filename,
                                                FILENAMES['variable_constants'])

        # Load graphs
        from ..util.serialization import ModelComponentDecoder
        graphs_list = load_json_from_zip(zip_filename, FILENAMES['graphs'],
                                           decoder=ModelComponentDecoder)
        graphs = FactorGraph.load_graphs(graphs_list)
        primary_model = graphs[0]
        secondary_graphs = graphs[1:]

        # { current_model_uuid : loaded_uuid}
        self._uuid_map = FactorGraph.reconcile_graphs(
            current_graphs=self.graphs,
            primary_previous_graph=primary_model,
            secondary_previous_graphs=secondary_graphs)

        new_parameters = InferenceParameters.load_parameters(
            uuid_map=self._uuid_map,
            mxnet_parameters=mxnet_parameters,
            variable_constants=variable_constants,
            mxnet_constants=mxnet_constants,
            current_params=self.params._params)
        self.params = new_parameters

        configuration = load_json_from_zip(zip_filename, FILENAMES['configuration'])
        self.load_configuration(configuration, self._uuid_map)

    def load_configuration(self, configuration, uuid_map):
        """
        Loads relevant inference configuration back from a file.
        Currently only loads the observed variables UUIDs back in,
        using the uuid_map parameter to store the
        correct current observed variables.

        :param config_file: The loaded configuration dictionary
        :type config_file: str
        :param uuid_map: A map of previous/loaded model component
         uuids to their current variable in the loaded graph.
        :type uuid_map: { current_model_uuid : loaded_previous_uuid}
        """
        pass
        # loaded_uuid = [uuid_map[uuid] for uuid in configuration['observed']]

    def get_serializable(self):
        """
        Returns the  mimimum set of properties that the object needs to save in order to be
        serialized down and loaded back in properly.
        :returns: A dictionary of configuration properties needed to serialize and reload this inference method.
        :rtypes: Dictionary that is JSON serializable.
        """
        return {'observed': self.observed_variable_UUIDs}

    def save(self, zip_filename=DEFAULT_ZIP):
        """
        Saves down everything needed to reload an inference algorithm.
        This method writes everything into a single zip archive, with 6 internal files.
        1. version.json - This has the version of serialization used to create the zip file.
        2. graphs.json - This is a networkx representation of all FactorGraphs used during Inference.
           See mxfusion.models.FactorGraph.save for more information.
        3. mxnet_parameters.npz - This is a numpy zip file saved using numpy.savez(), containing one file for each
           mxnet parameter in the InferenceParameters object. Each parameter is saved in a binary file named by the
           parameter's UUID.
        4. mxnet_constants.npz - The same as mxnet_parameters, except only for constant mxnet parameters.
        5. variable_constants.json - Parameters file of primitive data type constants, such as ints or floats.
           I.E. { UUID : int/float}
        6. configuration.json - This has other configuration related to inference such as the observation pattern.
        :param zip_filename: Path to and name of the zip archive to save the inference method as.
        :type zip_filename: str
        """
        # Retrieve dictionary representations of things to save
        mxnet_parameters, mxnet_constants, variable_constants = self.params.get_serializable()
        configuration = self.get_serializable()
        graphs = [g.as_json()for g in self._graphs]
        version_dict = {"serialization_version":
                                  SERIALIZATION_VERSION}

        files_to_save = []
        objects = [graphs, mxnet_parameters, mxnet_constants,
                   variable_constants, configuration, version_dict]
        ordered_filenames = [FILENAMES['graphs'], FILENAMES['mxnet_params'], FILENAMES['mxnet_constants'],
                             FILENAMES['variable_constants'], FILENAMES['configuration'], FILENAMES['version_file']]
        encodings = [ENCODINGS['json'], ENCODINGS['numpy'], ENCODINGS['numpy'],
                             ENCODINGS['json'], ENCODINGS['json'], ENCODINGS['json']]

        # Form each individual file buffer.
        for filename, obj, encoding in zip(ordered_filenames, objects, encodings):
            # For the FactorGraphs, configuration, and variable constants just write them as regular json.
            if encoding == ENCODINGS['json']:
                buffer = io.StringIO()
                json.dump(obj, buffer, ensure_ascii=False,
                          cls=ModelComponentEncoder)
            # For MXNet parameters, save them as numpy compressed zip files of arrays.
            # So a numpy-zip within the bigger zip.
            elif encoding == ENCODINGS['numpy']:
                buffer = io.BytesIO()
                np_obj = make_numpy(obj)
                np.savez(buffer, **np_obj)
            files_to_save.append((filename, buffer))

        # Form the overall zipfile stream
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a",
                             zipfile.ZIP_DEFLATED, False) as zip_file:
            for base_name, data in files_to_save:
                zip_file.writestr(base_name, data.getvalue())

        # Finally save the actual zipfile stream to disk
        with open(zip_filename, 'wb') as f:
            f.write(zip_buffer.getvalue())


class TransferInference(Inference):
    """
    The abstract Inference method for transferring the outcome of one inference
    method to another.

    :param inference_algorithm: The applied inference algorithm
    :type inference_algorithm: InferenceAlgorithm
    :param constants: Specify a list of model variables as constants
    :type constants: {Variable: mxnet.ndarray}
    :param hybridize: Whether to hybridize
     the MXNet Gluon block of the inference method.
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
